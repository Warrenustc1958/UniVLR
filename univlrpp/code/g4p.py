"""G4' (2026-11-12, dsw3, no training): training-free pointer + re-look on V*.
point: P0b layer-14 heads 8 11 7 13 (chosen in G3'), slot -> image-token attention renormalised over image tokens
       (sink excluded), mean over the 4 heads and 384 slots -> grid map; 3x3 smoothing; connected region >= 0.5 x peak
       around the peak cell -> bounding box -> expanded x2 (min 224 px, same rule as the G1' GT crop).
ask:   Qwen2.5-VL-7B-Instruct answers with the crop as second user image (same prompt / note as G1').
report: paired with G1' qwen rows (none / gt / full) on the same items.
phases: point|ask [--shard i --nshard n] | report
"""
import sys, json, math, argparse, collections
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import torch.nn.functional as F
import g123 as G

OUT = G.OUT / 'g4p'
HEADS = [8, 11, 7, 13]
LAYER = 14


class Stop(Exception):
    pass


def point(a):
    import p0
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as MQ
    OUT.mkdir(parents=True, exist_ok=True); fo = open(OUT / f'point{a.shard}.jsonl', 'w', buffering=1)
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
        ctx['p'] = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HD), -1).mean((0, 1))  # [n_img]
        raise Stop

    hk = layer.register_forward_pre_hook(hook, with_kwargs=True)
    items = G.vstar_items()[a.shard::a.nshard]
    for it in items:
        im = Image.open(it['img']).convert('RGB')
        t = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': it['text']}]}],
                                     add_generation_prompt=True) + p0.span(384)
        x = proc(text=[t], images=[im], return_tensors='pt'); zb, tb = p0.zimage(proc, im, 384)
        b = {'ids': x['input_ids'], 'pv': x['pixel_values'], 'thw': x['image_grid_thw'], 'zb': zb, 'tb': tb}
        _, hp, wp = [int(v) for v in b['thw'][0]]; gh, gw = hp // 2, wp // 2
        with torch.no_grad():
            ids, emb, pos = p0.embed(m, b, p0.vis(m, zb, tb))
            ctx.update(slots=(ids[0] == p0.SLOT).nonzero()[:, 0], img=(ids[0] == p0.IMG).nonzero()[:, 0])
            assert len(ctx['img']) == gh * gw
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
        sx, sy = wp * 14 / it['W'], hp * 14 / it['H']  # resized-image px per original px
        r0, r1 = min(r for r, _ in comp), max(r for r, _ in comp) + 1; c0, c1 = min(c for _, c in comp), max(c for _, c in comp) + 1
        pbox = [c0 * 28 / sx, r0 * 28 / sy, c1 * 28 / sx, r1 * 28 / sy]
        crop = G.ctx_box(pbox, it['W'], it['H']); gtc = G.ctx_box(it['box'], it['W'], it['H'])
        cx, cy = (pc + .5) * 28 / sx, (pr + .5) * 28 / sy; bx = it['box']
        gi = torch.zeros(gh, gw)
        for r in range(gh):
            for c in range(gw):
                if c * 28 < bx[2] * sx and (c + 1) * 28 > bx[0] * sx and r * 28 < bx[3] * sy and (r + 1) * 28 > bx[1] * sy:
                    gi[r, c] = 1
        row = dict(key=it['key'], cat=it['cat'], pbox=pbox, crop=crop, ncells=len(comp), grid=[gh, gw],
                   peak_in_gt=bool(bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]),
                   crop_contains_gt=bool(crop[0] <= bx[0] and crop[1] <= bx[1] and crop[2] >= bx[2] and crop[3] >= bx[3]),
                   iou_gtcrop=round(G.iou(crop, gtc), 4), mass_gt=round(float((P.cpu() * gi).sum()), 4),
                   base_gt=round(float(gi.mean()), 4), crop_frac=round((crop[2] - crop[0]) * (crop[3] - crop[1]) / (it['W'] * it['H']), 4))
        fo.write(json.dumps(row) + '\n')
    hk.remove(); print('POINT_DONE', a.shard, len(items), flush=True)


def ask(a):
    from PIL import Image
    from vlmeval.utils.matching_util import can_infer
    pts = {r['key']: r for f in sorted(OUT.glob('point*.jsonl')) for r in map(json.loads, open(f))}
    m, proc = G.load_teacher('qwen'); fo = open(OUT / f'ask{a.shard}.jsonl', 'w', buffering=1)
    items = [it for it in G.vstar_items() if it['key'] in pts][a.shard::a.nshard]
    for it in items:
        im = Image.open(it['img']).convert('RGB'); pt = pts[it['key']]
        letters = sorted(it['choices']); gi = letters.index(it['gold'])
        out, lp, n = G.ask(m, proc, [im, G.render(im, pt['crop'])], G.NOTE + it['text'], letters)
        fo.write(json.dumps(dict(key=it['key'], ok=can_infer(out, it['choices']) == it['gold'],
                                 p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4), out=out[:60])) + '\n')
    print('ASK_DONE', a.shard, len(items), flush=True)


def report(a):
    t = {r['key']: r for f in sorted((G.OUT / 'teach_qwen').glob('shard*.jsonl')) for r in map(json.loads, open(f)) if r['ds'] == 'vstar'}
    pts = {r['key']: r for f in sorted(OUT.glob('point*.jsonl')) for r in map(json.loads, open(f))}
    ak = {r['key']: r for f in sorted(OUT.glob('ask*.jsonl')) for r in map(json.loads, open(f))}
    keys = sorted(set(t) & set(ak)); acc = lambda f: 100 * np.mean([f(k) for k in keys])
    ptr = lambda k: int(ak[k]['ok']); cond = lambda c: (lambda k: int(t[k]['res'][c]['ok']))
    L = [f"# G4′ training-free pointer + re-look on V* (n={len(keys)}; pointer = P0b layer {LAYER} heads {HEADS}; answerer = Qwen2.5-VL-7B-Instruct)", '',
         '| condition | V* acc | Δ vs ptr [95% CI] | p_gold |', '|:--|--:|:--|--:|']
    for nm, f, pg in (('none (original only)', cond('none'), lambda k: t[k]['res']['none']['p_gold']),
                      ('full (whole image as 2nd image)', cond('full'), lambda k: t[k]['res']['full']['p_gold']),
                      ('**ptr (pointer crop)**', ptr, lambda k: ak[k]['p_gold']),
                      ('gt (GT crop, upper bound)', cond('gt'), lambda k: t[k]['res']['gt']['p_gold'])):
        d = G.boot([f(k) - ptr(k) for k in keys])
        L.append(f'| {nm} | {acc(f):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | {np.mean([pg(k) for k in keys]):.3f} |')
    for nm, cmp in (('ptr − none', cond('none')), ('ptr − full', cond('full')), ('ptr − gt', cond('gt'))):
        d = G.boot([ptr(k) - cmp(k) for k in keys]); L.append(f'\n{nm}: {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]')
    P = [pts[k] for k in keys]
    L += ['', '## pointer quality', '',
          f"- peak cell inside the GT box: {100 * np.mean([p['peak_in_gt'] for p in P]):.1f}%",
          f"- crop fully contains the GT box: {100 * np.mean([p['crop_contains_gt'] for p in P]):.1f}%",
          f"- IoU(pointer crop, GT crop): mean {np.mean([p['iou_gtcrop'] for p in P]):.2f}, median {np.median([p['iou_gtcrop'] for p in P]):.2f}",
          f"- attention mass in GT box / uniform baseline: {np.mean([p['mass_gt'] / max(p['base_gt'], 1e-6) for p in P]):.1f}",
          f"- crop area / image area: median {np.median([p['crop_frac'] for p in P]):.3f}; region cells median {np.median([p['ncells'] for p in P]):.0f}", '',
          '| subset | n | none | ptr | gt |', '|:--|--:|--:|--:|--:|']
    for nm, sel in (('crop contains GT box', lambda k: pts[k]['crop_contains_gt']), ('crop misses GT box', lambda k: not pts[k]['crop_contains_gt']),
                    ('direct_attributes', lambda k: pts[k]['cat'] == 'direct_attributes'), ('relative_position', lambda k: pts[k]['cat'] == 'relative_position')):
        ks = [k for k in keys if sel(k)]
        if ks:
            L.append(f"| {nm} | {len(ks)} | {100 * np.mean([cond('none')(k) for k in ks]):.1f} | {100 * np.mean([ptr(k) for k in ks]):.1f} | "
                     f"{100 * np.mean([cond('gt')(k) for k in ks]):.1f} |")
    (OUT / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['point', 'ask', 'report'])
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    a = p.parse_args()
    {'point': point, 'ask': ask, 'report': report}[a.phase](a)
