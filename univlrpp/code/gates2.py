"""Gate suite 2 (2026-11-12, dsw3, no training) for the latent-pointer + glimpse design
(UniVLR++_LatentPointer_Glimpse_Design_20261112.md).
ask : V* with Qwen2.5-VL-7B-Instruct in the NEW layout = one user turn [image, question prompt, note, crop image(s)] (crop after
      the question, as prefix reuse needs). conditions: none | gt | full_tok (whole image resized to the GT crop's pixel count)
      | rand (GT-crop-sized box, random position) | center (same size, centred) | ptr1 (P0b layer-14 pointer, single peak = G4')
      | ptr2 (union of the top-2 peak regions) | ptr2x (the two regions as two crops) | tbox (Qwen's own text bbox_2d -> crop).
g3q : does an untrained query localise? per model (qwen | p0b) two queries in one forward: lastq = last question token,
      slot1 = one placeholder <|univlr|> token appended after the question (inside the user turn). Attention mass in the GT box
      over image tokens / uniform, per layer and head; heads picked on even items, scored on odd.
unit: design unit tests on Qwen-Instruct (one V* item + GT crop): U1 incremental prefill (crop ViT run separately) == full
      prefill; U2 pointer bypass + cache crop == no-pointer path; U3 slot-only LoRA leaves the prefix KV identical;
      U4 prefix M-RoPE positions unchanged by the appended crop.
phases: ask [--shard --nshard] | g3q --model qwen|p0b | unit | report
"""
import os, sys, json, re, math, copy, random, argparse, collections
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import torch.nn.functional as F
import g123 as G

OUT = G.OUT / 'gates2'
NOTE1 = '\nZoomed-in view of a region of the image:'
NOTE2 = '\nZoomed-in views of two regions of the image:'
HEADS, PLAYER = [8, 11, 7, 13], 14
LAYERS = [4, 8, 14, 20, 27]
CONDS = ['none', 'gt', 'full_tok', 'rand', 'center', 'ptr1', 'ptr2', 'ptr2x', 'tbox']


def chat2(proc, parts):
    content, ims = [], []
    for t, v in parts:
        if t == 'image':
            content.append({'type': 'image'}); ims.append(v)
        else:
            content.append({'type': 'text', 'text': v})
    return proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True), ims


@torch.no_grad()
def ask2(m, proc, parts, letters=None, max_new=32):
    t, ims = chat2(proc, parts); x = proc(text=[t], images=ims, return_tensors='pt').to(m.device)
    g = m.generate(**x, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                   output_scores=True, return_dict_in_generate=True)
    out = proc.tokenizer.decode(g.sequences[0, x['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    lp = None
    if letters:
        ids = [proc.tokenizer.convert_tokens_to_ids(c) for c in letters]
        lp = torch.log_softmax(g.scores[0][0].float(), -1)[ids].tolist()
    return out, lp, x


class Stop(Exception):
    pass


def make_pointer():
    """P0b layer-14 heads 8 11 7 13: slot -> image attention (image-token softmax, mean over heads and 384 slots) -> grid map."""
    import p0
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as MQ
    ck = str(G.CKPT['p0b'])
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(ck, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': 0}).eval()
    proc = AutoProcessor.from_pretrained(ck, min_pixels=p0.MINP, max_pixels=p0.MAXP)
    proc.tokenizer.add_tokens(p0.TOKS, special_tokens=True)
    assert proc.tokenizer.convert_tokens_to_ids(p0.TOKS) == [p0.START, p0.SLOT, p0.LEND, p0.END]
    p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    lm = m.model.language_model; cfg = lm.config
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads; HD = cfg.hidden_size // NH; grp = NH // NKV
    MS = (getattr(lm.layers[0].self_attn, 'rope_scaling', None) or cfg.rope_scaling)['mrope_section']
    layer = lm.layers[PLAYER]; att = layer.self_attn; ctx = {}

    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']; cos, sin = kwargs['position_embeddings']
        x = layer.input_layernorm(h); n = x.shape[1]
        q = att.q_proj(x).view(1, n, NH, HD).transpose(1, 2); k = att.k_proj(x).view(1, n, NKV, HD).transpose(1, 2)
        q, k = MQ.apply_multimodal_rotary_pos_emb(q, k, cos, sin, MS)
        q = q[0, HEADS][:, ctx['slots']].float(); k = k[0, [hh // grp for hh in HEADS]][:, ctx['img']].float()
        ctx['p'] = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HD), -1).mean((0, 1))
        raise Stop

    layer.register_forward_pre_hook(hook, with_kwargs=True)

    @torch.no_grad()
    def locate(im, text):
        t = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
                                     add_generation_prompt=True) + p0.span(384)
        x = proc(text=[t], images=[im], return_tensors='pt'); zb, tb = p0.zimage(proc, im, 384)
        b = {'ids': x['input_ids'], 'pv': x['pixel_values'], 'thw': x['image_grid_thw'], 'zb': zb, 'tb': tb}
        _, hp, wp = [int(v) for v in b['thw'][0]]
        ids, emb, pos = p0.embed(m, b, p0.vis(m, zb, tb))
        ctx.update(slots=(ids[0] == p0.SLOT).nonzero()[:, 0], img=(ids[0] == p0.IMG).nonzero()[:, 0])
        try:
            lm(inputs_embeds=emb, position_ids=pos, attention_mask=p0.causal4(ids), use_cache=False)
        except Stop:
            pass
        gh, gw = hp // 2, wp // 2
        return ctx['p'].view(gh, gw).cpu(), wp * 14 / im.size[0], hp * 14 / im.size[1]
    return locate


def regions(P, sx, sy):
    """single peak region (>= 0.5 x peak, connected) and an optional second region outside the dilated first one."""
    gh, gw = P.shape
    S = F.avg_pool2d(P[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    keep = (S >= 0.5 * S.max()).numpy()

    def grow(r0, c0, ban):
        comp, stack, seen = [], [(r0, c0)], {(r0, c0)}
        while stack:
            r, c = stack.pop(); comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < gh and 0 <= cc < gw and keep[rr, cc] and not ban[rr, cc] and (rr, cc) not in seen:
                    seen.add((rr, cc)); stack.append((rr, cc))
        return comp

    box = lambda comp: [min(c for _, c in comp) * 28 / sx, min(r for r, _ in comp) * 28 / sy,
                        (max(c for _, c in comp) + 1) * 28 / sx, (max(r for r, _ in comp) + 1) * 28 / sy]
    pr, pc = divmod(int(S.argmax()), gw); none = np.zeros((gh, gw), bool)
    c1 = grow(pr, pc, none); ban = np.zeros((gh, gw), bool)
    for r, c in c1:
        ban[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    S2 = S.clone(); S2[torch.from_numpy(ban | ~keep)] = -1
    c2 = None
    if float(S2.max()) > 0:
        r2, cc2 = divmod(int(S2.argmax()), gw); c2 = grow(r2, cc2, ban)
    return box(c1), (box(c2) if c2 else None), float(S.max()), (float(S2.max()) if c2 else 0.0)


def union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def contains(c, b):
    return bool(c[0] <= b[0] and c[1] <= b[1] and c[2] >= b[2] and c[3] >= b[3])


def tbox_parse(out, thw, W, H):
    # numbers inside the first "bbox_2d": [...] list only (the old regex also caught the "2" of "2d", 11-12 bug)
    mm = re.search(r'bbox_2d"?\s*:\s*\[([^\]]+)\]', out) or re.search(r'\[\s*(-?\d[^\]]*)\]', out)
    nums = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', mm.group(1))[:4]] if mm else []
    if len(nums) < 4:
        return None
    rw, rh = int(thw[0][2]) * 14, int(thw[0][1]) * 14
    x1, y1, x2, y2 = nums[0] * W / rw, nums[1] * H / rh, nums[2] * W / rw, nums[3] * H / rh
    x1, x2 = sorted((max(0, min(W, x1)), max(0, min(W, x2)))); y1, y2 = sorted((max(0, min(H, y1)), max(0, min(H, y2))))
    return [x1, y1, x2, y2] if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def ask_phase(a):
    from PIL import Image
    from vlmeval.dataset import build_dataset
    from vlmeval.utils.matching_util import can_infer
    od = OUT / ('smoke' if a.limit else 'ask'); od.mkdir(parents=True, exist_ok=True); fp = od / f'shard{a.shard}.jsonl'
    done = {json.loads(l)['key'] for l in open(fp)} if fp.exists() else set()
    rawq = {int(r['index']): str(r['question']) for _, r in build_dataset('VStarBench').data.iterrows()}
    its = [it for it in G.vstar_items()[a.shard::a.nshard] if it['key'] not in done]
    if a.limit:
        its = its[:a.limit]
    locate = make_pointer(); m, proc = G.load_teacher('qwen'); fo = open(fp, 'a', buffering=1)
    for it in its:
        im = Image.open(it['img']).convert('RGB'); W, H = it['W'], it['H']; bx = it['box']
        gtc = G.ctx_box(bx, W, H); gw_, gh_ = gtc[2] - gtc[0], gtc[3] - gtc[1]
        rng = random.Random(it['key']); rx, ry = rng.uniform(0, W - gw_), rng.uniform(0, H - gh_)
        rand = [rx, ry, rx + gw_, ry + gh_]; cen = [(W - gw_) / 2, (H - gh_) / 2, (W + gw_) / 2, (H + gh_) / 2]
        P, sx, sy = locate(im, it['text']); b1, b2, s1, s2 = regions(P, sx, sy)
        p1 = G.ctx_box(b1, W, H); p2 = G.ctx_box(union(b1, b2), W, H) if b2 else p1; p2b = G.ctx_box(b2, W, H) if b2 else None
        tq = rawq[it['key']] + '\nLocate the region of the image that is needed to answer this question. ' \
             'Output its bounding box in JSON format as {"bbox_2d": [x1, y1, x2, y2]}.'
        tout, _, tx = ask2(m, proc, [('image', im), ('text', tq)], None, max_new=64)
        tb = tbox_parse(tout, tx['image_grid_thw'].cpu(), W, H); tbc = G.ctx_box(tb, W, H) if tb else None
        gr = G.render(im, gtc); A = gr.size[0] * gr.size[1]; s = math.sqrt(A / (W * H))
        ftok = im.resize((max(28, int(W * s)), max(28, int(H * s))), Image.BICUBIC)
        crops = {'gt': gtc, 'rand': rand, 'center': cen, 'ptr1': p1, 'ptr2': p2, 'tbox': tbc}
        base = [('image', im), ('text', it['text'])]
        parts = {'none': base, 'full_tok': base + [('text', NOTE1), ('image', ftok)],
                 'ptr2x': base + ([('text', NOTE2), ('image', G.render(im, p1)), ('image', G.render(im, p2b))] if p2b
                                  else [('text', NOTE1), ('image', G.render(im, p1))]),
                 'tbox': base + [('text', NOTE1), ('image', G.render(im, tbc))] if tbc else base}
        for c in ('gt', 'rand', 'center', 'ptr1', 'ptr2'):
            parts[c] = base + [('text', NOTE1), ('image', G.render(im, crops[c]))]
        letters = sorted(it['choices']); gi = letters.index(it['gold']); res = {}
        for c in CONDS:
            out, lp, x = ask2(m, proc, parts[c], letters)
            r = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4),
                     out=out[:40], ntok=int(x['input_ids'].shape[1]))
            if crops.get(c) is not None:
                r.update(iou=round(G.iou(crops[c], gtc), 4), contains=contains(crops[c], bx))
            res[c] = r
        fo.write(json.dumps(dict(key=it['key'], cat=it['cat'], second=b2 is not None, s1=round(s1, 5), s2=round(s2, 5),
                                 tbox_ok=tb is not None, tbox_out=tout[:80],
                                 ptr2x_contains=bool(contains(p1, bx) or (p2b is not None and contains(p2b, bx))), res=res)) + '\n')
        print('ASK', it['key'], {c: int(v['ok']) for c, v in res.items()}, flush=True)
    print('ASK_DONE', a.shard, len(its), flush=True)


CONDS3 = ['none', 'gt_rep', 'ptr1_rep', 'tbox_new', 'tbox_rep', 'tbox_old']


def ask3_phase(a):
    """layout follow-up: rep = [image, question, note, crop, question again]; old = [image, crop, note + question]
    (G1'/G4' layout); tbox = Qwen's own bbox_2d (parser fixed)."""
    from PIL import Image
    from vlmeval.dataset import build_dataset
    from vlmeval.utils.matching_util import can_infer
    od = OUT / ('smoke3' if a.limit else 'ask3'); od.mkdir(parents=True, exist_ok=True); fp = od / f'shard{a.shard}.jsonl'
    done = {json.loads(l)['key'] for l in open(fp)} if fp.exists() else set()
    rawq = {int(r['index']): str(r['question']) for _, r in build_dataset('VStarBench').data.iterrows()}
    its = [it for it in G.vstar_items()[a.shard::a.nshard] if it['key'] not in done]
    if a.limit:
        its = its[:a.limit]
    locate = make_pointer(); m, proc = G.load_teacher('qwen'); fo = open(fp, 'a', buffering=1)
    for it in its:
        im = Image.open(it['img']).convert('RGB'); W, H = it['W'], it['H']; bx = it['box']; gtc = G.ctx_box(bx, W, H)
        P_, sx, sy = locate(im, it['text']); b1 = regions(P_, sx, sy)[0]; p1 = G.ctx_box(b1, W, H)
        tq = rawq[it['key']] + '\nLocate the region of the image that is needed to answer this question. ' \
             'Output its bounding box in JSON format as {"bbox_2d": [x1, y1, x2, y2]}.'
        tout, _, tx = ask2(m, proc, [('image', im), ('text', tq)], None, max_new=160)
        tb = tbox_parse(tout, tx['image_grid_thw'].cpu(), W, H); tbc = G.ctx_box(tb, W, H) if tb else None
        Q = it['text']; base = [('image', im), ('text', Q)]
        rep_ = lambda c: base + [('text', NOTE1), ('image', G.render(im, c)), ('text', '\n' + Q)]
        parts = {'none': base, 'gt_rep': rep_(gtc), 'ptr1_rep': rep_(p1),
                 'tbox_new': base + [('text', NOTE1), ('image', G.render(im, tbc))] if tbc else base,
                 'tbox_rep': rep_(tbc) if tbc else base,
                 'tbox_old': [('image', im), ('image', G.render(im, tbc)), ('text', G.NOTE + Q)] if tbc else base}
        crops = {'gt_rep': gtc, 'ptr1_rep': p1, 'tbox_new': tbc, 'tbox_rep': tbc, 'tbox_old': tbc}
        letters = sorted(it['choices']); gi = letters.index(it['gold']); res = {}
        for c in CONDS3:
            out, lp, x = ask2(m, proc, parts[c], letters)
            r = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4),
                     ntok=int(x['input_ids'].shape[1]))
            if crops.get(c) is not None:
                r.update(iou=round(G.iou(crops[c], gtc), 4), contains=contains(crops[c], bx))
            res[c] = r
        fo.write(json.dumps(dict(key=it['key'], cat=it['cat'], tbox_ok=tb is not None, tbox=tb, tbox_out=tout[:80],
                                 tbox_raw_iou=round(G.iou(tb, bx), 4) if tb else None, res=res)) + '\n')
        print('ASK3', it['key'], {c: int(v['ok']) for c, v in res.items()}, flush=True)
    print('ASK3_DONE', a.shard, len(its), flush=True)


def g3q_phase(a):
    import p0
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as MQ
    od = OUT / ('smoke' if a.limit else 'g3q'); od.mkdir(parents=True, exist_ok=True); fo = open(od / f'{a.model}.jsonl', 'w', buffering=1)
    if a.model == 'qwen':
        proc = p0.processor(); path = str(p0.QWEN)
    else:
        path = str(G.CKPT['p0b']); proc = AutoProcessor.from_pretrained(path, min_pixels=p0.MINP, max_pixels=p0.MAXP)
        proc.tokenizer.add_tokens(p0.TOKS, special_tokens=True)
    assert proc.tokenizer.convert_tokens_to_ids(p0.TOKS) == [p0.START, p0.SLOT, p0.LEND, p0.END]
    p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': 0}).eval()
    if a.model == 'qwen':  # untrained padding rows of Qwen-Instruct -> mean row (as P0)
        with torch.no_grad():
            for W_ in (m.model.get_input_embeddings().weight, m.lm_head.weight):
                W_[p0.START:p0.END + 1] = W_[:p0.START].float().mean(0).to(W_.dtype)
    lm = m.model.language_model; cfg = lm.config
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads; HD = cfg.hidden_size // NH; grp = NH // NKV
    MS = (getattr(lm.layers[0].self_attn, 'rope_scaling', None) or cfg.rope_scaling)['mrope_section']
    ctx, store = {}, {}

    def make(L):
        layer = lm.layers[L]; att = layer.self_attn

        def hook(mod, args, kwargs):
            h = args[0] if args else kwargs['hidden_states']; cos, sin = kwargs['position_embeddings']
            x = layer.input_layernorm(h); n = x.shape[1]
            q = att.q_proj(x).view(1, n, NH, HD).transpose(1, 2); k = att.k_proj(x).view(1, n, NKV, HD).transpose(1, 2)
            q, k = MQ.apply_multimodal_rotary_pos_emb(q, k, cos, sin, MS)
            k = k[0].repeat_interleave(grp, dim=0)[:, ctx['img']].float()
            rec = {}
            for nm, qp in ctx['q'].items():
                qh = q[0][:, qp].float()  # [NH, HD]
                pi = torch.softmax(torch.einsum('hd,hnd->hn', qh, k) / math.sqrt(HD), -1)  # [NH, n_img], head h vs its own key
                rec[nm] = (pi * ctx['inside']).sum(-1).tolist()
            store[L] = rec
        return hook

    hs = [lm.layers[L].register_forward_pre_hook(make(L), with_kwargs=True) for L in LAYERS]
    items = [dict(ds='vstar', **x) for x in G.vstar_items()] + [dict(ds='vcot', **x) for x in G.vcot_items() if x['box'] is not None]
    if a.limit:
        items = items[:a.limit]
    for it in items:
        im = Image.open(it['img']).convert('RGB'); text = it['text'] if it['ds'] == 'vstar' else it['q']
        t = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
                                     add_generation_prompt=True)
        i0 = t.index('<|im_end|>', t.index('<|vision_end|>')); t = t[:i0] + '<|univlr|>' + t[i0:]
        x = proc(text=[t], images=[im], return_tensors='pt')
        b = {'ids': x['input_ids'], 'pv': x['pixel_values'], 'thw': x['image_grid_thw']}
        _, hp, wp = [int(v) for v in b['thw'][0]]; gh, gw = hp // 2, wp // 2
        sx, sy = wp * 14 / it['W'], hp * 14 / it['H']; bx = [it['box'][0] * sx, it['box'][1] * sy, it['box'][2] * sx, it['box'][3] * sy]
        r = torch.arange(gh)[:, None]; c = torch.arange(gw)[None, :]
        inside = ((c * 28 < bx[2]) & ((c + 1) * 28 > bx[0]) & (r * 28 < bx[3]) & ((r + 1) * 28 > bx[1])).flatten().float()
        with torch.no_grad():
            ids, emb, pos = p0.embed(m, b, None)
            sp = int((ids[0] == p0.SLOT).nonzero()[0, 0])
            ctx.update(img=(ids[0] == p0.IMG).nonzero()[:, 0], inside=inside.to(m.device), q={'lastq': sp - 1, 'slot1': sp}); store.clear()
            lm(inputs_embeds=emb, position_ids=pos, attention_mask=p0.causal4(ids), use_cache=False)
        fo.write(json.dumps(dict(ds=it['ds'], key=it['key'], base=float(inside.mean()), layers={str(k): v for k, v in store.items()})) + '\n')
    for h in hs:
        h.remove()
    print('G3Q_DONE', a.model, len(items), flush=True)


def kvs(c, i):
    if hasattr(c, 'layers'):
        return c.layers[i].keys, c.layers[i].values
    return c.key_cache[i], c.value_cache[i]


@torch.no_grad()
def unit_phase(a):
    import p0
    from PIL import Image
    from transformers import DynamicCache, Qwen2_5_VLForConditionalGeneration
    OUT.mkdir(parents=True, exist_ok=True)
    proc = p0.processor(); p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(p0.QWEN), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': 0}).eval()
    lm = m.model.language_model; lp = lambda x: F.log_softmax(x.float(), -1); R_ = {}
    it = G.vstar_items()[0]; im = Image.open(it['img']).convert('RGB'); crop = G.render(im, G.ctx_box(it['box'], it['W'], it['H']))
    t_full, ims = chat2(proc, [('image', im), ('text', it['text']), ('text', NOTE1), ('image', crop)])
    x = proc(text=[t_full + it['gold'] + '<|im_end|>'], images=ims, return_tensors='pt')
    a0 = proc(text=[t_full], images=ims, return_tensors='pt')['input_ids'].shape[1]
    ids = x['input_ids'].to(m.device); thw = x['image_grid_thw'].to(m.device); pv = x['pixel_values']; n = ids.shape[1]
    vs = proc.tokenizer.convert_tokens_to_ids('<|vision_start|>'); c0 = int((ids[0] == vs).nonzero()[1, 0])
    # full references
    ref = m(input_ids=ids, pixel_values=pv.to(m.device, torch.bfloat16), image_grid_thw=thw, attention_mask=torch.ones_like(ids),
            use_cache=False).logits[0, a0 - 1:n - 1].float()
    b = {'ids': ids.cpu(), 'pv': pv, 'thw': x['image_grid_thw']}
    _, emb, pos = p0.embed(m, b, None); c4 = p0.causal4(ids)
    own = m.lm_head(lm(inputs_embeds=emb, position_ids=pos, attention_mask=c4, use_cache=False).last_hidden_state[0, a0 - 1:n - 1]).float()
    # crop through the ViT on its own (deployment), spliced into the embedding sequence
    n0 = int(x['image_grid_thw'][0].prod()); fj = p0.vis(m, pv, x['image_grid_thw']); fc = p0.vis(m, pv[n0:], x['image_grid_thw'][1:])
    R_['vit_joint_vs_separate_maxdiff'] = float((fj[fj.shape[0] - fc.shape[0]:].float() - fc.float()).abs().max())
    emb_s = emb.clone(); ipos = (ids[0] == p0.IMG).nonzero()[:, 0]; emb_s[0, ipos[-fc.shape[0]:]] = fc.to(emb.dtype)
    # U4: prefix positions computed on the prefix alone == positions inside the full sequence
    pos_pre, _ = m.model.get_rope_index(ids[:, :c0], thw[:1], None)
    R_['U4_prefix_rope_equal'] = bool(torch.equal(pos_pre, pos[:, :, :c0]))
    # U1: incremental prefill (prefix, then crop segment + answer with the prefix cache as past)
    o1 = lm(inputs_embeds=emb_s[:, :c0], position_ids=pos[:, :, :c0], attention_mask=c4[:, :, :c0, :c0], past_key_values=DynamicCache(), use_cache=True)
    cache_ptr = copy.deepcopy(o1.past_key_values)
    o2 = lm(inputs_embeds=emb_s[:, c0:], position_ids=pos[:, :, c0:], attention_mask=c4[:, :, c0:n, :n], past_key_values=o1.past_key_values, use_cache=True)
    inc = m.lm_head(o2.last_hidden_state[0, a0 - 1 - c0:n - 1 - c0]).float()
    y = ids[0, a0:n]
    R_['U1_kl_inc_vs_ownfull'] = float(F.kl_div(lp(inc), lp(own), log_target=True, reduction='batchmean'))
    R_['U1_kl_inc_vs_hf'] = float(F.kl_div(lp(inc), lp(ref), log_target=True, reduction='batchmean'))
    R_['U1_argmax_agree_hf'] = float((inc.argmax(-1) == ref.argmax(-1)).float().mean())
    R_['U1_ce_inc_hf'] = [float(F.cross_entropy(inc, y)), float(F.cross_entropy(ref, y))]
    # U2: pointer bypass: append one slot to the prefix cache, read it, crop the cache back to the prefix, continue
    es = m.model.get_input_embeddings()(torch.tensor([[p0.SLOT]], device=m.device))
    lm(inputs_embeds=es, position_ids=pos[:, :, c0 - 1:c0] + 1, attention_mask=torch.ones(1, 1, 1, c0 + 1, dtype=torch.bool, device=m.device),
       past_key_values=cache_ptr, use_cache=True)
    cache_ptr.crop(c0)
    o3 = lm(inputs_embeds=emb_s[:, c0:], position_ids=pos[:, :, c0:], attention_mask=c4[:, :, c0:n, :n], past_key_values=cache_ptr, use_cache=True)
    byp = m.lm_head(o3.last_hidden_state[0, a0 - 1 - c0:n - 1 - c0]).float()
    R_['U2_maxdiff_bypass_vs_nopointer'] = float((byp - inc).abs().max())
    # U3: slot-only LoRA (random, all linear layers of every decoder layer) must leave the prefix KV identical
    gen = torch.Generator(device='cpu').manual_seed(0); lora = {}; flag = {'on': False, 'mask': None}

    def add(mod):
        A = (torch.randn(8, mod.in_features, generator=gen) * 0.02).to(m.device, torch.bfloat16)
        B = (torch.randn(mod.out_features, 8, generator=gen) * 0.02).to(m.device, torch.bfloat16)
        def hk(md, inp, out):
            if not flag['on']:
                return out
            return out + (inp[0] @ A.T @ B.T) * flag['mask']
        return mod.register_forward_hook(hk)
    hooks = []
    for L in lm.layers:
        for mod in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj, L.self_attn.o_proj, L.mlp.gate_proj, L.mlp.up_proj, L.mlp.down_proj):
            hooks.append(add(mod))
    e_ps = torch.cat([emb[:, :c0], es], 1); p_ps = torch.cat([pos[:, :, :c0], pos[:, :, c0 - 1:c0] + 1], 2)
    m_ps = torch.ones(c0 + 1, c0 + 1, dtype=torch.bool, device=m.device).tril()[None, None]
    flag['mask'] = torch.zeros(1, c0 + 1, 1, device=m.device, dtype=torch.bfloat16); flag['mask'][0, -1] = 1
    res = {}
    for on in (False, True):
        flag['on'] = on
        o = lm(inputs_embeds=e_ps, position_ids=p_ps, attention_mask=m_ps, past_key_values=DynamicCache(), use_cache=True)
        res[on] = (o.past_key_values, o.last_hidden_state[0, -1].float())
    for h in hooks:
        h.remove()
    d = 0.0
    for i in range(len(lm.layers)):
        (k0, v0), (k1, v1) = kvs(res[False][0], i), kvs(res[True][0], i)
        d = max(d, float((k0[:, :, :c0] - k1[:, :, :c0]).abs().max()), float((v0[:, :, :c0] - v1[:, :, :c0]).abs().max()))
    R_['U3_prefix_kv_maxdiff_lora_on_off'] = d
    R_['U3_slot_hidden_change'] = float((res[True][1] - res[False][1]).abs().max())
    R_['meta'] = dict(key=it['key'], n=n, c0=c0, a0=a0)
    R_['PASS'] = dict(U1=R_['U1_kl_inc_vs_ownfull'] < 1e-3, U2=R_['U2_maxdiff_bypass_vs_nopointer'] == 0.0,
                      U3=R_['U3_prefix_kv_maxdiff_lora_on_off'] == 0.0 and R_['U3_slot_hidden_change'] > 0, U4=R_['U4_prefix_rope_equal'])
    json.dump(R_, open(OUT / 'unit.json', 'w'), indent=1); print('UNIT', json.dumps(R_), flush=True); print('UNIT_DONE', flush=True)


def report(a):
    L = ['# Gate suite 2 (2026-11-12, dsw3, no training)', '']
    rows = [json.loads(l) for f in sorted((OUT / 'ask').glob('shard*.jsonl')) for l in open(f)]
    if rows:
        acc = lambda c, rs=rows: 100 * np.mean([r['res'][c]['ok'] for r in rs])
        L += [f'## G1″ + text bbox + multi-peak: V* (n={len(rows)}), answerer Qwen2.5-VL-7B-Instruct, NEW layout '
              '(crop after the question in the same user turn)', '',
              '| condition | acc | Δ vs none [95% CI] | Δ vs gt [95% CI] | p_gold | IoU to GT crop | crop ⊇ GT box | prompt tokens |',
              '|:--|--:|:--|:--|--:|--:|--:|--:|']
        for c in CONDS:
            d1 = G.boot([int(r['res'][c]['ok']) - int(r['res']['none']['ok']) for r in rows])
            d2 = G.boot([int(r['res'][c]['ok']) - int(r['res']['gt']['ok']) for r in rows])
            io = [r['res'][c]['iou'] for r in rows if 'iou' in r['res'][c]]
            ct = [r['res'][c]['contains'] for r in rows if 'contains' in r['res'][c]]
            if c == 'ptr2x':
                ct = [r['ptr2x_contains'] for r in rows]
            L.append(f"| {c} | {acc(c):.1f} | {100 * d1[0]:+.1f} [{100 * d1[1]:+.1f}, {100 * d1[2]:+.1f}] | "
                     f"{100 * d2[0]:+.1f} [{100 * d2[1]:+.1f}, {100 * d2[2]:+.1f}] | {np.mean([r['res'][c]['p_gold'] for r in rows]):.3f} | "
                     f"{np.mean(io) if io else float('nan'):.2f} | {100 * np.mean(ct) if ct else float('nan'):.0f}% | "
                     f"{np.mean([r['res'][c]['ntok'] for r in rows]):.0f} |")
        L += ['', f"text bbox parsed: {100 * np.mean([r['tbox_ok'] for r in rows]):.0f}%  (unparsed -> answered without a crop); "
                  f"second pointer region found: {100 * np.mean([r['second'] for r in rows]):.0f}%", '',
              '| category | n | ' + ' | '.join(CONDS) + ' |', '|:--|--:|' + '--:|' * len(CONDS)]
        for cat in sorted({r['cat'] for r in rows}):
            rs = [r for r in rows if r['cat'] == cat]
            L.append(f'| {cat} | {len(rs)} | ' + ' | '.join(f'{acc(c, rs):.1f}' for c in CONDS) + ' |')
        L += ['', 'Old layout reference (G1′/G4′, crop before the question): none 75.4 / full 80.6 / ptr1 86.9 / gt 92.1.', '']
    for mdl in ('qwen', 'p0b'):
        f = OUT / 'g3q' / f'{mdl}.jsonl'
        if not f.exists():
            continue
        rs = [json.loads(l) for l in open(f)]
        L += [f'## G3″ untrained-query localisation: {mdl} (n={len(rs)}; ratio = mass in GT box / uniform, 1 = chance)', '',
              '| query | layer | all heads | top-4 heads (even->odd) [95% CI] | heads |', '|:--|--:|--:|:--|:--|']
        # rows were stored as [NH, NH] (query of head i against key of head j, broadcast bug in g3q_phase);
        # the diagonal is the correct per-head value
        per_head = lambda v: np.diag(np.array(v)) if np.array(v).ndim == 2 else np.array(v)
        for q in ('lastq', 'slot1'):
            for Ly in map(str, LAYERS):
                rat = np.array([per_head(r['layers'][Ly][q]) / max(r['base'], 1e-6) for r in rs])
                top = np.argsort(-rat[0::2].mean(0))[:4]; t4 = G.boot(rat[1::2][:, top].mean(1))
                L.append(f'| {q} | {Ly} | {rat.mean():.2f} | {t4[0]:.2f} [{t4[1]:.2f}, {t4[2]:.2f}] | {top.tolist()} |')
        L.append('')
    r3 = [json.loads(l) for f in sorted((OUT / 'ask3').glob('shard*.jsonl')) for l in open(f)]
    if r3:
        t = {r['key']: r for f in sorted((G.OUT / 'teach_qwen').glob('shard*.jsonl')) for r in map(json.loads, open(f)) if r['ds'] == 'vstar'}
        g4 = {r['key']: r for f in sorted((G.OUT / 'g4p').glob('ask*.jsonl')) for r in map(json.loads, open(f))}
        ok = lambda r, c: int(r['res'][c]['ok'])
        L += [f'## Layout follow-up + text bbox (parser fixed): V* (n={len(r3)}), answerer Qwen2.5-VL-7B-Instruct', '',
              'rep = [image, question, note, crop, question again]; new = [image, question, note, crop]; '
              'old = [image, crop, note + question] (G1′/G4′)', '',
              '| condition | acc | Δ vs none [95% CI] | p_gold | IoU to GT crop | crop ⊇ GT box |', '|:--|--:|:--|--:|--:|--:|']
        for c in CONDS3:
            d = G.boot([ok(r, c) - ok(r, 'none') for r in r3])
            io = [r['res'][c]['iou'] for r in r3 if 'iou' in r['res'][c]]
            ct = [r['res'][c]['contains'] for r in r3 if 'contains' in r['res'][c]]
            L.append(f"| {c} | {100 * np.mean([ok(r, c) for r in r3]):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | "
                     f"{np.mean([r['res'][c]['p_gold'] for r in r3]):.3f} | {np.mean(io) if io else float('nan'):.2f} | "
                     f"{100 * np.mean(ct) if ct else float('nan'):.0f}% |")
        for nm, src, f in (('gt_old (G1′)', t, lambda r: int(t[r['key']]['res']['gt']['ok'])),
                           ('full_old (G1′)', t, lambda r: int(t[r['key']]['res']['full']['ok'])),
                           ('ptr1_old (G4′)', g4, lambda r: int(g4[r['key']]['ok']))):
            if all(r['key'] in src for r in r3):
                d = G.boot([f(r) - ok(r, 'none') for r in r3])
                L.append(f"| {nm} | {100 * np.mean([f(r) for r in r3]):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | | | |")
        tbi = [r['tbox_raw_iou'] for r in r3 if r['tbox_raw_iou'] is not None]
        L += ['', f"text bbox parsed {100 * np.mean([r['tbox_ok'] for r in r3]):.0f}%; IoU(text box, GT box) over parsed: "
                  f"mean {np.mean(tbi) if tbi else float('nan'):.2f}, median {np.median(tbi) if tbi else float('nan'):.2f}, "
                  f"IoU >= 0.5 {100 * np.mean([v >= 0.5 for v in tbi]) if tbi else float('nan'):.0f}%", '',
              '| category | n | ' + ' | '.join(CONDS3) + ' |', '|:--|--:|' + '--:|' * len(CONDS3)]
        for cat in sorted({r['cat'] for r in r3}):
            rs = [r for r in r3 if r['cat'] == cat]
            L.append(f'| {cat} | {len(rs)} | ' + ' | '.join(f"{100 * np.mean([ok(r, c) for r in rs]):.1f}" for c in CONDS3) + ' |')
        L.append('')
    if (OUT / 'unit.json').exists():
        U = json.load(open(OUT / 'unit.json'))
        L += ['## Unit tests (Qwen2.5-VL-7B-Instruct, one V* item + GT crop, new layout)', '', '```', json.dumps(U, indent=1), '```', '']
    g0 = G.OUT / 'g0p'
    for ds in ('HRBench4K', 'MME-RealWorld-Lite'):
        rs = [json.loads(l) for f in sorted(g0.glob(f'{ds}_shard*.jsonl')) for l in open(f)]
        if not rs:
            continue
        L += [f'## G0′ official UniVLR-Stage1 latent intervention: {ds} (n={len(rs)}; official wrapper, univlr 1 step, 5120 tokens)', '',
              '| latent fed | acc | Δ vs head [95% CI] | answers changed vs head | mean latent vectors fed per item |', '|:--|--:|:--|--:|--:|']
        for md in ('head', 'zero', 'noise', 'swap'):
            d = G.boot([int(r['res'][md]['ok']) - int(r['res']['head']['ok']) for r in rs])
            fl = np.mean([r['res'][md]['pred'] != r['res']['head']['pred'] for r in rs])
            L.append(f"| {md} | {100 * np.mean([r['res'][md]['ok'] for r in rs]):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | "
                     f"{100 * fl:.1f}% | {np.mean([r['res'][md]['n_feed'] for r in rs]):.1f} |")
        L.append('')
    (OUT / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['ask', 'ask3', 'g3q', 'unit', 'report'])
    p.add_argument('--model', choices=['qwen', 'p0b'], default='qwen')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    {'ask': ask_phase, 'ask3': ask3_phase, 'g3q': g3q_phase, 'unit': unit_phase, 'report': report}[a.phase](a)
