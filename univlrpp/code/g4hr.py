"""G4'-HR (2026-11-12, dsw4, no training): training-free pointer + re-look on HRBench4K (no bbox -> accuracy only).
Pointer = P0b layer-14 heads 8 11 7 13 (same rule as g4p.py); answerer = Qwen2.5-VL-7B-Instruct, conditions:
none (original only) | full (whole image as second image) | ptr (pointer crop from the original 4K pixels).
Official VLMEvalKit prompt; per-row accuracy (not the cyclic HRBench average). phases: run --shard i --nshard n | report
"""
import sys, json, math, argparse, collections
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import torch.nn.functional as F
import g123 as G

OUT = G.OUT / 'g4hr'
HEADS, LAYER, DS = [8, 11, 7, 13], 14, 'HRBench4K'


def items():
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

    ds = build_dataset(DS); pb = PB(); pb.set_dump_image(ds.dump_image); out = []
    for _, line in ds.data.iterrows():
        msg = pb.build_prompt(line, dataset=DS)
        ip = [x['value'] for x in msg if x['type'] == 'image'][0]
        ch = {c: str(line[c]) for c in 'ABCD' if c in line and not (isinstance(line[c], float) and math.isnan(line[c]))}
        W, H = Image.open(ip).size
        out.append(dict(key=int(line['index']), img=ip, W=W, H=H, gold=str(line['answer']), choices=ch,
                        cat=str(line.get('category', '')), text='\n'.join(x['value'] for x in msg if x['type'] == 'text')))
    return out


class Stop(Exception):
    pass


def pointer_model():
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
    layer = lm.layers[LAYER]; att = layer.self_attn; ctx = {}

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
        _, hp, wp = [int(v) for v in b['thw'][0]]; gh, gw = hp // 2, wp // 2
        ids, emb, pos = p0.embed(m, b, p0.vis(m, zb, tb))
        ctx.update(slots=(ids[0] == p0.SLOT).nonzero()[:, 0], img=(ids[0] == p0.IMG).nonzero()[:, 0])
        try:
            lm(inputs_embeds=emb, position_ids=pos, attention_mask=p0.causal4(ids), use_cache=False)
        except Stop:
            pass
        P = ctx['p'].view(gh, gw)
        S = F.avg_pool2d(P[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
        pr, pc = divmod(int(S.argmax()), gw); keep = (S >= 0.5 * S[pr, pc]).cpu().numpy()
        comp, stack, seen = [], [(pr, pc)], {(pr, pc)}
        while stack:
            r, c = stack.pop(); comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < gh and 0 <= cc < gw and keep[rr, cc] and (rr, cc) not in seen:
                    seen.add((rr, cc)); stack.append((rr, cc))
        sx, sy = wp * 14 / im.size[0], hp * 14 / im.size[1]
        r0, r1 = min(r for r, _ in comp), max(r for r, _ in comp) + 1; c0, c1 = min(c for _, c in comp), max(c for _, c in comp) + 1
        return [c0 * 28 / sx, r0 * 28 / sy, c1 * 28 / sx, r1 * 28 / sy], len(comp), (gh, gw)
    return locate


def run(a):
    from PIL import Image
    from vlmeval.utils.matching_util import can_infer
    OUT.mkdir(parents=True, exist_ok=True); fp = OUT / f'shard{a.shard}.jsonl'
    done = {json.loads(l)['key'] for l in open(fp)} if fp.exists() else set()
    its = [it for it in items()[a.shard::a.nshard] if it['key'] not in done]
    if a.limit:
        its = its[:a.limit]
    locate = pointer_model(); m, proc = G.load_teacher('qwen'); fo = open(fp, 'a', buffering=1)
    for it in its:
        im = Image.open(it['img']).convert('RGB'); pbox, nc, grid = locate(im, it['text'])
        crop = G.ctx_box(pbox, it['W'], it['H']); letters = sorted(it['choices']); gi = letters.index(it['gold'])
        res = {}
        for c, ims, txt in (('none', [im], it['text']), ('full', [im, G.render(im)], G.NOTE + it['text']),
                            ('ptr', [im, G.render(im, crop)], G.NOTE + it['text'])):
            out, lp, n = G.ask(m, proc, ims, txt, letters)
            res[c] = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4),
                          out=out[:50], ntok=n)
        fo.write(json.dumps(dict(key=it['key'], cat=it['cat'], crop=[round(v) for v in crop], ncells=nc, grid=grid,
                                 crop_frac=round((crop[2] - crop[0]) * (crop[3] - crop[1]) / (it['W'] * it['H']), 4), res=res)) + '\n')
    print('HR_DONE', a.shard, len(its), flush=True)


def report(a):
    rows = [json.loads(l) for f in sorted(OUT.glob('shard*.jsonl')) for l in open(f)]
    acc = lambda rs, c: 100 * np.mean([r['res'][c]['ok'] for r in rs])
    L = [f'# G4′-HR: training-free pointer + re-look on {DS} (n={len(rows)} rows; pointer = P0b layer {LAYER} heads {HEADS}; '
         'answerer = Qwen2.5-VL-7B-Instruct; per-row accuracy, not the cyclic HRBench average)', '',
         '| subset | n | none | full | ptr | ptr − none [95% CI] | ptr − full [95% CI] |', '|:--|--:|--:|--:|--:|:--|:--|']
    for nm, rs in [('all', rows)] + sorted(((f'category {k}', [r for r in rows if r['cat'] == k]) for k in {r['cat'] for r in rows}), key=lambda x: x[0]):
        d1 = G.boot([int(r['res']['ptr']['ok']) - int(r['res']['none']['ok']) for r in rs])
        d2 = G.boot([int(r['res']['ptr']['ok']) - int(r['res']['full']['ok']) for r in rs])
        L.append(f"| {nm} | {len(rs)} | {acc(rs, 'none'):.1f} | {acc(rs, 'full'):.1f} | {acc(rs, 'ptr'):.1f} | "
                 f"{100 * d1[0]:+.1f} [{100 * d1[1]:+.1f}, {100 * d1[2]:+.1f}] | {100 * d2[0]:+.1f} [{100 * d2[1]:+.1f}, {100 * d2[2]:+.1f}] |")
    L += ['', f"p_gold none / full / ptr: {np.mean([r['res']['none']['p_gold'] for r in rows]):.3f} / "
              f"{np.mean([r['res']['full']['p_gold'] for r in rows]):.3f} / {np.mean([r['res']['ptr']['p_gold'] for r in rows]):.3f}",
          f"crop area / image area median {np.median([r['crop_frac'] for r in rows]):.4f}; region cells median {np.median([r['ncells'] for r in rows]):.0f}"]
    (OUT / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['run', 'report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    {'run': run, 'report': report}[a.phase](a)
