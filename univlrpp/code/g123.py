"""G1' / G2' / G3' gates (2026-11-11, dsw3, no training) for the latent-pointer + re-look plan.
G1': does a teacher that also sees the GT region (second user image = high-res crop around the GT bbox) beat the same
     model without it?  V* (official bbox, Datasets/vstar_ann) with the official VLMEvalKit prompt; Visual-CoT holdout
     (P0 holdout, crop = the auxiliary image) by answer CE. Placebo: second image = the full image.
G2': does the teacher degrade smoothly as the crop drifts from the GT region (IoU 0.7 / 0.5 / 0.3 / 0)?
G3': do P0 / P0b slot attentions (recomputed from q/k at layers 4 8 14 20 27) put their mass on the GT box?
     Sink handling: renormalise over image tokens only; also excluding high-norm image tokens (> 5x median);
     heads selected on even items, evaluated on odd items; baseline = fraction of image cells inside the box.
phases: teach --teacher qwen|s1|s2 [--shard i --nshard n] | g3 --ckpt p0|p0b | report
"""
import os, sys, json, math, glob, argparse, collections
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import torch.nn.functional as F

R = Path('/mnt/cpfs/jieduan/univlr')
OUT = R / 'experiments/p0_20261111/gates'
VA = R / 'Datasets/vstar_ann/vstar_bench'
MON = R / 'Datasets/Monet-SFT-125K'
P0D = R / 'experiments/p0_20261111/data/holdout.json'
TEACH = {'qwen': R / 'Models/Qwen2.5-VL-7B-Instruct', 's1': R / 'Models/UniVLR-Stage1-7B-converted',
         's2': R / 'Models/UniVLR-Stage2-7B-converted'}
CKPT = {'p0': R / 'experiments/p0_20261111/full/ckpt', 'p0b': R / 'experiments/p0_20261111/full_b/ckpt'}
MAXP, MINP = 5120 * 28 * 28, 4 * 28 * 28
NOTE = 'The second image is a zoomed-in view of a region of the first image.\n'
IOUS = [0.7, 0.5, 0.3, 0.0]
LAYERS = [4, 8, 14, 20, 27]
NVCOT = 200


# ---------------------------------------------------------------- data
def vstar_items():
    """V* rows with the official prompt (Qwen2VLPromptMixin) and the GT box in dumped-image pixels (union of targets)."""
    from PIL import Image
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.base import BaseModel
    from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin

    class PB(Qwen2VLPromptMixin, BaseModel):
        INSTALL_REQ = False
        INTERLEAVE = True

        def __init__(self):
            super().__init__(use_custom_prompt=True)

        def generate_inner(self, message, dataset=None):
            raise NotImplementedError

    ds = build_dataset('VStarBench'); pb = PB(); pb.set_dump_image(ds.dump_image)
    # vstar_ann images are git-LFS pointers; the official [x,y,w,h] boxes fit the TSV images unscaled (190/191 inside,
    # max x2/W 1.02, checked 11-11), so they are used as TSV pixel coordinates, clamped to the image.
    ann = {}
    for f in glob.glob(str(VA / '*' / '*.json')):
        j = json.load(open(f)); ann[str(j['question']).strip()] = j['bbox']
    items, miss = [], 0
    for _, line in ds.data.iterrows():
        a = ann.get(str(line['question']).strip())
        if a is None:
            miss += 1; continue
        msg = pb.build_prompt(line, dataset='VStarBench')
        ip = [x['value'] for x in msg if x['type'] == 'image'][0]
        txt = '\n'.join(x['value'] for x in msg if x['type'] == 'text')
        W, H = Image.open(ip).size
        bs = [[max(0, x), max(0, y), min(W, x + w), min(H, y + h)] for x, y, w, h in a]
        box = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
        ch = {c: str(line[c]) for c in 'ABCD' if c in line and not (isinstance(line[c], float) and math.isnan(line[c]))}
        items.append(dict(key=int(line['index']), img=ip, text=txt, box=box, W=W, H=H, choices=ch, gold=str(line['answer']),
                          cat=str(line['category']), ntarget=len(bs)))
    print('VSTAR items', len(items), 'unmatched', miss, flush=True)
    return items


def tmatch(o, c):
    import cv2
    O = np.asarray(o.convert('L'), np.float32); C = np.asarray(c.convert('L'), np.float32)
    if C.shape[0] > O.shape[0] or C.shape[1] > O.shape[1]:
        return -1.0, None
    r = cv2.matchTemplate(O, C, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
    return float(mx), [loc[0], loc[1], loc[0] + C.shape[1], loc[1] + C.shape[0]]


def vcot_items():
    from PIL import Image
    out = []
    for s in json.load(open(P0D))[:NVCOT]:
        o, c = Image.open(MON / s['img']), Image.open(MON / s['crop'])
        sc, box = tmatch(o, c)
        out.append(dict(key=s['key'], img=str(MON / s['img']), crop=str(MON / s['crop']), q=s['q'], a=s['a'],
                        box=box if sc >= 0.9 else None, W=o.size[0], H=o.size[1], match=sc))
    print('VCOT items', len(out), 'bbox matched', sum(x['box'] is not None for x in out), flush=True)
    return out


# ---------------------------------------------------------------- crops
def ctx_box(b, W, H, mn=224, f=2.0):
    """GT box expanded x f around its centre (at least mn px per side), kept inside the image."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w = min(W, max(f * (b[2] - b[0]), mn)); h = min(H, max(f * (b[3] - b[1]), mn))
    x1 = min(max(0, cx - w / 2), W - w); y1 = min(max(0, cy - h / 2), H - h)
    return [x1, y1, x1 + w, y1 + h]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def shift_box(c, W, H, t):
    """same-size box shifted so that IoU(c, shifted) = t (horizontal first, then vertical); clamped if needed."""
    w, h = c[2] - c[0], c[3] - c[1]
    dx = w * (1 - t) / (1 + t) if t > 0 else w; dy = h * (1 - t) / (1 + t) if t > 0 else h
    for ddx, ddy in ((dx, 0), (-dx, 0), (0, dy), (0, -dy)):
        x1, y1 = c[0] + ddx, c[1] + ddy
        if x1 >= 0 and y1 >= 0 and x1 + w <= W and y1 + h <= H:
            return [x1, y1, x1 + w, y1 + h]
    best = None
    for ddx, ddy in ((dx, 0), (-dx, 0), (0, dy), (0, -dy)):
        x1 = min(max(0, c[0] + ddx), W - w); y1 = min(max(0, c[1] + ddy), H - h); nb = [x1, y1, x1 + w, y1 + h]
        if best is None or abs(iou(c, nb) - t) < abs(iou(c, best) - t):
            best = nb
    return best


def render(im, box=None, min_side_px=448):
    """crop (or the whole image) upscaled so its area is at least min_side_px^2 (detail for the ViT patch grid)."""
    from PIL import Image
    c = im.crop(tuple(int(round(v)) for v in box)) if box is not None else im
    a = c.size[0] * c.size[1]; s = max(1.0, math.sqrt(min_side_px ** 2 / max(a, 1)))
    return c.resize((max(28, int(c.size[0] * s)), max(28, int(c.size[1] * s))), Image.BICUBIC) if s > 1 else c


# ---------------------------------------------------------------- teacher (G1' / G2')
def load_teacher(name):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    p = str(TEACH[name])
    m, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(p, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                 device_map={'': 0}, output_loading_info=True)
    miss = list(info.get('missing_keys', [])); unexp = list(info.get('unexpected_keys', []))
    print('LOAD', name, 'missing', len(miss), miss[:4], 'unexpected', len(unexp), unexp[:4], flush=True)
    assert not miss, f'{name}: {len(miss)} missing keys'
    proc = AutoProcessor.from_pretrained(p, min_pixels=MINP, max_pixels=MAXP)
    return m.eval(), proc


def chat(proc, ims, text):
    content = [{'type': 'image'} for _ in ims] + [{'type': 'text', 'text': text}]
    return proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True)


@torch.no_grad()
def ask(m, proc, ims, text, letters):
    t = chat(proc, ims, text); x = proc(text=[t], images=ims, return_tensors='pt').to(m.device)
    g = m.generate(**x, max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None,
                   output_scores=True, return_dict_in_generate=True)
    out = proc.tokenizer.decode(g.sequences[0, x['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    ids = [proc.tokenizer.convert_tokens_to_ids(c) for c in letters]
    lp = torch.log_softmax(g.scores[0][0].float(), -1)[ids]
    return out, lp.tolist(), int(x['input_ids'].shape[1])


@torch.no_grad()
def ans_ce(m, proc, ims, q, a):
    t = chat(proc, ims, q)
    a0 = proc(text=[t], images=ims, return_tensors='pt')['input_ids'].shape[1]
    x = proc(text=[t + a + '<|im_end|>'], images=ims, return_tensors='pt').to(m.device)
    h = m.model(input_ids=x['input_ids'], pixel_values=x['pixel_values'], image_grid_thw=x['image_grid_thw'],
                attention_mask=x['attention_mask'], use_cache=False).last_hidden_state
    lg = m.lm_head(h[0, a0 - 1:-1]).float(); y = x['input_ids'][0, a0:]
    return float(F.cross_entropy(lg, y))


def teach(a):
    from PIL import Image
    from vlmeval.utils.matching_util import can_infer
    od = OUT / f'teach_{a.teacher}'; od.mkdir(parents=True, exist_ok=True)
    fo_p = od / f'shard{a.shard}.jsonl'
    done = {(r['ds'], r['key']) for r in map(json.loads, open(fo_p))} if fo_p.exists() else set()
    m, proc = load_teacher(a.teacher)
    vs = vstar_items()[a.shard::a.nshard]; vc = vcot_items()[a.shard::a.nshard]
    if a.limit:
        vs, vc = vs[:a.limit], vc[:a.limit]
    fo = open(fo_p, 'a', buffering=1)
    for it in vs:
        if ('vstar', it['key']) in done:
            continue
        im = Image.open(it['img']).convert('RGB'); gt = ctx_box(it['box'], it['W'], it['H'])
        conds = {'none': ([im], it['text'], 1.0), 'gt': ([im, render(im, gt)], NOTE + it['text'], 1.0),
                 'full': ([im, render(im)], NOTE + it['text'], None)}
        for t in IOUS:
            sb = shift_box(gt, it['W'], it['H'], t); conds[f'iou{t}'] = ([im, render(im, sb)], NOTE + it['text'], iou(gt, sb))
        letters = sorted(it['choices']); gi = letters.index(it['gold'])
        res = {}
        for c, (ims, txt, io) in conds.items():
            out, lp, n = ask(m, proc, ims, txt, letters)
            pg = float(torch.softmax(torch.tensor(lp), 0)[gi])
            res[c] = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(pg, 4), out=out[:60], iou=io, ntok=n)
        fo.write(json.dumps({'ds': 'vstar', 'key': it['key'], 'cat': it['cat'], 'res': res}, ensure_ascii=False) + '\n')
        print('V*', it['key'], {c: int(v['ok']) for c, v in res.items()}, flush=True)
    for it in vc:
        if ('vcot', it['key']) in done:
            continue
        im = Image.open(it['img']).convert('RGB'); cr = Image.open(it['crop']).convert('RGB')
        res = {'none': ans_ce(m, proc, [im], it['q'], it['a']),
               'gt': ans_ce(m, proc, [im, render(cr)], NOTE + it['q'], it['a']),
               'full': ans_ce(m, proc, [im, render(im)], NOTE + it['q'], it['a'])}
        fo.write(json.dumps({'ds': 'vcot', 'key': it['key'], 'res': {k: round(v, 4) for k, v in res.items()}}) + '\n')
    print('TEACH_DONE', a.teacher, a.shard, flush=True)


# ---------------------------------------------------------------- G3': slot attention localisation
def g3(a):
    import p0
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as MQ
    od = OUT / f'g3_{a.ckpt}'; od.mkdir(parents=True, exist_ok=True); fo = open(od / 'rows.jsonl', 'w', buffering=1)
    ck = str(CKPT[a.ckpt])
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(ck, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': 0}).eval()
    proc = AutoProcessor.from_pretrained(ck, min_pixels=p0.MINP, max_pixels=p0.MAXP)
    proc.tokenizer.add_tokens(p0.TOKS, special_tokens=True)
    assert proc.tokenizer.convert_tokens_to_ids(p0.TOKS) == [p0.START, p0.SLOT, p0.LEND, p0.END]
    p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    lm = m.model.language_model; cfg = lm.config
    NH, NKV = cfg.num_attention_heads, cfg.num_key_value_heads; HD = cfg.hidden_size // NH
    MS = (getattr(lm.layers[0].self_attn, 'rope_scaling', None) or cfg.rope_scaling)['mrope_section']
    ctx, store = {}, {}

    def make(L):
        layer = lm.layers[L]; att = layer.self_attn

        def hook(mod, args, kwargs):
            h = args[0] if args else kwargs['hidden_states']; cos, sin = kwargs['position_embeddings']
            x = layer.input_layernorm(h); n = x.shape[1]
            q = att.q_proj(x).view(1, n, NH, HD).transpose(1, 2); k = att.k_proj(x).view(1, n, NKV, HD).transpose(1, 2)
            q, k = MQ.apply_multimodal_rotary_pos_emb(q, k, cos, sin, MS)
            k = k.repeat_interleave(NH // NKV, dim=1)
            sl, im, inside = ctx['slots'], ctx['img'], ctx['inside']
            lg = (q[0][:, sl].float() @ k[0].float().transpose(-1, -2)) / math.sqrt(HD)  # [NH, S, n]
            causal = torch.arange(n, device=lg.device)[None, :] <= sl[:, None]
            lg = lg.masked_fill(~causal[None], float('-inf')); pf = torch.softmax(lg, -1)
            li = lg[:, :, im]; pi = torch.softmax(li, -1)
            hn = h[0, im].float().norm(dim=-1); keep = hn <= 5 * hn.median()
            pk = torch.softmax(li[:, :, keep], -1)
            top1 = pf.argmax(-1).flatten().tolist()
            rec = dict(mass=(pi * inside).sum(-1).mean(-1).tolist(), mass_k=(pk * inside[keep]).sum(-1).mean(-1).tolist(),
                       share_img=pf[:, :, im].sum(-1).mean(-1).tolist(), n_outlier=int((~keep).sum()),
                       top1=collections.Counter(top1).most_common(3))
            if ctx.get('check') and L == 14:  # rebuild the attention output at the slot rows, compare with the module
                v = att.v_proj(x).view(1, n, NKV, HD).transpose(1, 2).repeat_interleave(NH // NKV, dim=1)
                o = (pf.to(v.dtype) @ v[0]).transpose(0, 1).reshape(len(sl), NH * HD)
                ctx['manual14'] = att.o_proj(o).float()
            store[L] = rec
        return hook

    hs = [lm.layers[L].register_forward_pre_hook(make(L), with_kwargs=True) for L in LAYERS]
    cap = {}
    hs.append(lm.layers[14].self_attn.register_forward_hook(lambda mod, inp, out: cap.__setitem__('o14', out[0])))
    items = [dict(ds='vstar', **x) for x in vstar_items()] + [dict(ds='vcot', **x) for x in vcot_items() if x['box'] is not None]
    if a.limit:
        items = items[:a.limit] + [x for x in items if x['ds'] == 'vcot'][:a.limit]
    for i, it in enumerate(items):
        im = Image.open(it['img']).convert('RGB'); text = it['text'] if it['ds'] == 'vstar' else it['q']
        t = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
                                     add_generation_prompt=True) + p0.span(384)
        x = proc(text=[t], images=[im], return_tensors='pt'); zb, tb = p0.zimage(proc, im, 384)
        b = {'ids': x['input_ids'], 'pv': x['pixel_values'], 'thw': x['image_grid_thw'], 'zb': zb, 'tb': tb}
        _, hp, wp = [int(v) for v in b['thw'][0]]; gh, gw = hp // 2, wp // 2
        sx, sy = wp * 14 / it['W'], hp * 14 / it['H']; bx = [it['box'][0] * sx, it['box'][1] * sy, it['box'][2] * sx, it['box'][3] * sy]
        r = torch.arange(gh)[:, None]; c = torch.arange(gw)[None, :]
        inside = ((c * 28 < bx[2]) & ((c + 1) * 28 > bx[0]) & (r * 28 < bx[3]) & ((r + 1) * 28 > bx[1])).flatten().float()
        with torch.no_grad():
            ids, emb, pos = p0.embed(m, b, p0.vis(m, zb, tb))
            img = (ids[0] == p0.IMG).nonzero()[:, 0]; sl = (ids[0] == p0.SLOT).nonzero()[:, 0]
            assert len(img) == gh * gw, (len(img), gh, gw)
            ctx.update(slots=sl, img=img, inside=inside.to(m.device), check=(i == 0)); store.clear()
            lm(inputs_embeds=emb, position_ids=pos, attention_mask=p0.causal4(ids), use_cache=False)
        row = dict(ds=it['ds'], key=it['key'], base=float(inside.mean()), ngrid=gh * gw, layers={str(k): v for k, v in store.items()})
        if i == 0:
            mo = cap['o14'][0, sl].float(); d = float((ctx['manual14'] - mo).abs().max() / mo.abs().max())
            row['check_rel_maxdiff_L14'] = d; print('G3_CHECK rel maxdiff layer14', round(d, 4), flush=True)
            assert d < 0.05, d
        fo.write(json.dumps(row) + '\n')
    for h in hs:
        h.remove()
    print('G3_DONE', a.ckpt, len(items), flush=True)


# ---------------------------------------------------------------- report
def boot(x, n=4000):
    x = np.asarray(x, float)
    if not len(x):
        return float('nan'), float('nan'), float('nan')
    b = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def report(a):
    L = ['# G1′ / G2′ / G3′ gates (2026-11-11, dsw3, no training)', '',
         'G1′/G2′: teacher = model as released, loaded as Qwen2.5-VL; V* official VLMEvalKit prompt, greedy, can_infer; '
         'second user image = GT region expanded x2 (min 224 px), upscaled to >= 448² px; placebo `full` = whole image as '
         'second image; `iouT` = same-size crop shifted to IoU T with the GT crop. p_gold = softmax over option letters at '
         'the first generated token. Visual-CoT: answer CE (lower = better), P0 holdout first 200.', '']
    CONDS = ['none', 'gt', 'full'] + [f'iou{t}' for t in IOUS]
    for t in TEACH:
        rows = [json.loads(l) for f in sorted((OUT / f'teach_{t}').glob('shard*.jsonl')) for l in open(f)] if (OUT / f'teach_{t}').exists() else []
        vs = [r for r in rows if r['ds'] == 'vstar']; vc = [r for r in rows if r['ds'] == 'vcot']
        if not rows:
            continue
        L += [f'## teacher {t}  (V* n={len(vs)}, Visual-CoT n={len(vc)})', '',
              '| cond | V* acc | Δacc vs none [95% CI] | V* p_gold | mean IoU to GT crop | Visual-CoT CE |', '|:--|--:|:--|--:|--:|--:|']
        for c in CONDS:
            acc = [r['res'][c]['ok'] for r in vs if c in r['res']]
            d = boot([int(r['res'][c]['ok']) - int(r['res']['none']['ok']) for r in vs if c in r['res']])
            pg = np.mean([r['res'][c]['p_gold'] for r in vs if c in r['res']]) if acc else float('nan')
            io = [r['res'][c]['iou'] for r in vs if c in r['res'] and r['res'][c]['iou'] is not None]
            ce = np.mean([r['res'][c] for r in vc if c in r['res']]) if any(c in r['res'] for r in vc) else float('nan')
            L.append(f"| {c} | {100 * np.mean(acc):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | {pg:.3f} | "
                     f"{np.mean(io) if io else float('nan'):.2f} | {ce:.3f} |")
        seq = ['gt'] + [f'iou{t_}' for t_ in IOUS]
        pgs = [np.mean([r['res'][c]['p_gold'] for r in vs]) for c in seq]
        L += ['', f"G2′ p_gold along IoU 1 → 0.7 → 0.5 → 0.3 → 0: {' → '.join(f'{v:.3f}' for v in pgs)}; "
                  f"monotone non-increasing: {all(pgs[i] >= pgs[i + 1] - 1e-9 for i in range(len(pgs) - 1))}", '']
    for ck in CKPT:
        f = OUT / f'g3_{ck}' / 'rows.jsonl'
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        L += [f'## G3′ slot-attention localisation: {ck}  (n={len(rows)}; ratio = mass in GT box / uniform baseline, 1 = chance)', '',
              '| layer | all heads | top-4 heads (chosen on even items, scored on odd) [95% CI] | top-4, high-norm tokens excluded | '
              'share of full attention on image tokens | mean high-norm image tokens | most frequent top-1 key |', '|:--|--:|:--|--:|--:|--:|:--|']
        if 'check_rel_maxdiff_L14' in rows[0]:
            L.insert(len(L) - 2, f"(recomputed attention output vs module at layer 14, relative max diff: {rows[0]['check_rel_maxdiff_L14']:.4f})")
        for Ly in map(str, LAYERS):
            rat = np.array([np.array(r['layers'][Ly]['mass']) / max(r['base'], 1e-6) for r in rows])
            ratk = np.array([np.array(r['layers'][Ly]['mass_k']) / max(r['base'], 1e-6) for r in rows])
            ev, odd = rat[0::2], rat[1::2]; top = np.argsort(-ev.mean(0))[:4]
            t4 = boot(odd[:, top].mean(1)); t4k = ratk[1::2][:, top].mean()
            sh = np.mean([np.mean(r['layers'][Ly]['share_img']) for r in rows])
            no = np.mean([r['layers'][Ly]['n_outlier'] for r in rows])
            tops = collections.Counter(k for r in rows for k, _ in r['layers'][Ly]['top1'][:1]).most_common(2)
            L.append(f'| {Ly} | {rat.mean():.2f} | {t4[0]:.2f} [{t4[1]:.2f}, {t4[2]:.2f}] (heads {top.tolist()}) | {t4k:.2f} | {sh:.3f} | {no:.1f} | {tops} |')
        L.append('')
    (OUT / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['teach', 'g3', 'report'])
    p.add_argument('--teacher', choices=list(TEACH), default='qwen')
    p.add_argument('--ckpt', choices=list(CKPT), default='p0b')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if a.limit:
        OUT = OUT / 'smoke'
    {'teach': teach, 'g3': g3, 'report': report}[a.phase](a)
