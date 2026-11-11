"""Fair text-bbox baseline (2026-11-12, dsw4, no training). Answerer Qwen2.5-VL-7B-Instruct, layout [image, crop, note + question]
(the chosen design layout). Conditions on V* (191) and HRBench4K (800):
none | gt (V* only) | tbox (Qwen's own bbox_2d; one stricter re-prompt if unparsed, then no crop) | ptr_p0b (G4' pointer, P0b
layer-14 heads 8 11 7 13) | ptr_q0 (Stage B init = untrained Qwen pointer, heads 8 11 7 15). Same crop rule for all (ctx_box x2).
phases: run --ds vstar|hr4k --shard --nshard | report
"""
import sys, json, math, argparse
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import g123 as G
import gates2 as G2

OUT = G.OUT / 'tbox2'
LOC = ('\nLocate the region of the image that is needed to answer this question. '
       'Output its bounding box in JSON format as {"bbox_2d": [x1, y1, x2, y2]}.')
LOC2 = ('\nOutput exactly one JSON object {"bbox_2d": [x1, y1, x2, y2]} with integer pixel coordinates of the region needed '
        'to answer this question, and nothing else.')


def run(a):
    import ptrB
    from PIL import Image
    from vlmeval.dataset import build_dataset
    from vlmeval.utils.matching_util import can_infer
    od = OUT / ('smoke' if a.limit else ''); od.mkdir(parents=True, exist_ok=True); fp = od / f'{a.ds}_shard{a.shard}.jsonl'
    done = {json.loads(l)['key'] for l in open(fp)} if fp.exists() else set()
    if a.ds == 'vstar':
        its = G.vstar_items(); rawq = {int(r['index']): str(r['question']) for _, r in build_dataset('VStarBench').data.iterrows()}
    else:
        import g4hr
        its = g4hr.items(); rawq = {int(r['index']): str(r['question']) for _, r in build_dataset('HRBench4K').data.iterrows()}
    its = [it for it in its[a.shard::a.nshard] if it['key'] not in done]
    if a.limit:
        its = its[:a.limit]
    loc_p0b = G2.make_pointer()
    m, proc = ptrB.load_base(0); eng = ptrB.Engine(m, ptrB.make_ptr(m, 'cuda:0'))
    fo = open(fp, 'a', buffering=1)
    for it in its:
        im = Image.open(it['img']).convert('RGB'); W, H = it['W'], it['H']; crops = {}
        P, sx, sy = loc_p0b(im, it['text']); crops['ptr_p0b'] = G.ctx_box(G2.regions(P, sx, sy)[0], W, H)
        x = proc(text=[ptrB.prefix_text(proc, it['text'])], images=[im], return_tensors='pt')
        with torch.no_grad():
            st = eng.prefix(dict(ids=x['input_ids'], pv=x['pixel_values'], thw=x['image_grid_thw']))
            crops['ptr_q0'] = ptrB.pointer_crop(eng.slot(st).float(), st, W, H)
        tb, tries, touts = None, 0, []
        for suf in (LOC, LOC2):
            tries += 1; tout, _, tx = G2.ask2(m, proc, [('image', im), ('text', rawq[it['key']] + suf)], None, max_new=160)
            touts.append(tout[:80]); tb = G2.tbox_parse(tout, tx['image_grid_thw'].cpu(), W, H)
            if tb:
                break
        if tb:
            crops['tbox'] = G.ctx_box(tb, W, H)
        if a.ds == 'vstar':
            crops['gt'] = G.ctx_box(it['box'], W, H)
        letters = sorted(it['choices']); gi = letters.index(it['gold']); res = {}
        conds = [('none', [im], it['text'])] + [(c, [im, G.render(im, crops[c])], G.NOTE + it['text']) for c in ('gt', 'tbox', 'ptr_p0b', 'ptr_q0') if c in crops]
        if 'tbox' not in crops:
            conds.append(('tbox', [im], it['text']))  # unparsed after the retry -> no crop
        for c, ims, txt in conds:
            out, lp, _ = G.ask(m, proc, ims, txt, letters)
            r = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4))
            if a.ds == 'vstar' and c in crops:
                r.update(contains=G2.contains(crops[c], it['box']), iou=round(G.iou(crops[c], G.ctx_box(it['box'], W, H)), 4))
            res[c] = r
        fo.write(json.dumps(dict(key=it['key'], cat=it['cat'], tbox_ok=tb is not None, tbox_tries=tries, tbox_out=touts,
                                 tbox_raw_iou=round(G.iou(tb, it['box']), 4) if (tb and a.ds == 'vstar') else None, res=res)) + '\n')
        print('TB', a.ds, it['key'], {c: int(v['ok']) for c, v in res.items()}, flush=True)
    print('TB_DONE', a.ds, a.shard, len(its), flush=True)


def report(a):
    L = ['# Fair text-bbox baseline + G0′ on Stage2 (2026-11-12, dsw4)', '',
         'Answerer Qwen2.5-VL-7B-Instruct; layout [image, crop, note + question]; per-row accuracy. tbox = Qwen bbox_2d with one '
         're-prompt if unparsed (still unparsed -> answered without crop). ptr_p0b = G4′ pointer; ptr_q0 = untrained Qwen pointer '
         '(Stage B init). Same crop rule (x2 context, min 224 px) for every crop.', '']
    for ds in ('vstar', 'hr4k'):
        rs = [json.loads(l) for f in sorted(OUT.glob(f'{ds}_shard*.jsonl')) for l in open(f)]
        if not rs:
            continue
        conds = [c for c in ('none', 'gt', 'tbox', 'ptr_p0b', 'ptr_q0') if c in rs[0]['res']]
        ok = lambda r, c: int(r['res'][c]['ok'])
        L += [f'## {ds} (n={len(rs)})', '', '| condition | acc | Δ vs none [95% CI] | p_gold | IoU to GT crop | crop ⊇ GT box |', '|:--|--:|:--|--:|--:|--:|']
        for c in conds:
            d = G.boot([ok(r, c) - ok(r, 'none') for r in rs])
            io = [r['res'][c]['iou'] for r in rs if 'iou' in r['res'][c]]; ct = [r['res'][c]['contains'] for r in rs if 'contains' in r['res'][c]]
            L.append(f"| {c} | {100 * np.mean([ok(r, c) for r in rs]):.1f} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] | "
                     f"{np.mean([r['res'][c]['p_gold'] for r in rs]):.3f} | {np.mean(io) if io else float('nan'):.2f} | "
                     f"{100 * np.mean(ct) if ct else float('nan'):.0f}% |")
        L += ['', '| paired | Δ [95% CI] |', '|:--|:--|']
        for x_, y_ in (('ptr_p0b', 'tbox'), ('ptr_q0', 'tbox'), ('ptr_p0b', 'ptr_q0')):
            d = G.boot([ok(r, x_) - ok(r, y_) for r in rs]); L.append(f'| {x_} − {y_} | {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] |')
        t1 = np.mean([r['tbox_ok'] and r['tbox_tries'] == 1 for r in rs]); t2 = np.mean([r['tbox_ok'] for r in rs])
        L += ['', f'text bbox parsed: first prompt {100 * t1:.0f}%, after re-prompt {100 * t2:.0f}%']
        ti = [r['tbox_raw_iou'] for r in rs if r.get('tbox_raw_iou') is not None]
        if ti:
            L.append(f'IoU(text box, GT box): mean {np.mean(ti):.2f}, median {np.median(ti):.2f}, >= 0.5 {100 * np.mean([v >= 0.5 for v in ti]):.0f}%')
        L += ['', '| category | n | ' + ' | '.join(conds) + ' |', '|:--|--:|' + '--:|' * len(conds)]
        for cat in sorted({r['cat'] for r in rs}):
            rr = [r for r in rs if r['cat'] == cat]
            L.append(f'| {cat} | {len(rr)} | ' + ' | '.join(f'{100 * np.mean([ok(r, c) for r in rr]):.1f}' for c in conds) + ' |')
        L.append('')
    g0 = G.OUT / 'g0p'
    for ds in ('HRBench4K', 'MME-RealWorld-Lite'):
        rs = [json.loads(l) for f in sorted(g0.glob(f's2_{ds}_shard*.jsonl')) for l in open(f)]
        if not rs:
            continue
        L += [f'## G0′ official UniVLR-Stage2 latent intervention: {ds} (n={len(rs)}; official wrapper, steps 12, 5120 tokens)', '',
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
    p.add_argument('phase', choices=['run', 'report'])
    p.add_argument('--ds', choices=['vstar', 'hr4k'], default='vstar')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    {'run': run, 'report': report}[a.phase](a)
