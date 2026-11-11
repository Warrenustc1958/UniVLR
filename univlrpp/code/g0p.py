"""G0' (2026-11-12, dsw3, no training): does the official UniVLR-Stage1 latent carry information at inference?
Official wrapper Qwen2VLChatUniVLR with the s1_official config (Stage1-converted, univlr 1 step, max_new_tokens 512,
max_pixels 4,014,080). The released latent feed (mp._univlr_feed_vector) is wrapped in this process only (shared tree
untouched). Per item 4 runs: head (released behaviour; the fed latent vectors are recorded) | zero | noise (Gaussian,
per-vector norm matched) | swap (the previous item's recorded vectors, in order). Per-row accuracy + answers changed.
usage: python g0p.py --ds HRBench4K|MME-RealWorld-Lite --shard i --nshard n [--limit k]
"""
import os, sys, json, re, math, string, argparse
from pathlib import Path
R = '/mnt/cpfs/jieduan/univlr'; K = R + '/UniVLRplus/VLMEvalKit'
sys.path[:0] = [K, R + '/UniVLRplus']
os.environ.setdefault('UNIVLR_ROOT', R + '/UniVLRplus')
import torch

OUT = Path(R + '/experiments/p0_20261111/gates/g0p')
ST = {'mode': 'head', 'cur': [], 'prev': [], 'i': 0, 'n': 0}


def main(a):
    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.qwen2_vl.univlr import Qwen2VLChatUniVLR
    from vlmeval.utils.matching_util import can_infer
    import src.train.monkey_patch_forward_univlr_inference as mp
    orig = mp._univlr_feed_vector

    def feed(model, h):
        v = orig(model, h); ST['n'] += 1; md = ST['mode']
        if md == 'head':
            ST['cur'].append(v.detach().float().cpu()); return v
        if md == 'zero':
            return torch.zeros_like(v)
        if md == 'noise':
            z = torch.randn_like(v.float()); return (z * v.float().norm(dim=-1, keepdim=True) / z.norm(dim=-1, keepdim=True)).to(v.dtype)
        src = ST['prev']
        if not src:
            return torch.zeros_like(v)
        w = src[ST['i'] % len(src)].to(v.device, v.dtype); ST['i'] += 1
        return w if w.shape == v.shape else torch.zeros_like(v)
    mp._univlr_feed_vector = feed

    od = OUT / ('smoke' if a.limit else ''); od.mkdir(parents=True, exist_ok=True)
    fp = od / (f'{a.ds}_shard{a.shard}.jsonl' if a.model == 's1' else f'{a.model}_{a.ds}_shard{a.shard}.jsonl')
    # s1 = s1_official (univlr, 1 step: emits an EMPTY latent block, probe 11-12); s2 = univlr_s2_1105 (steps, 12 latents)
    cfg = {'s1': ('UniVLR-Stage1-7B-converted', 'univlr', 1), 's2': ('UniVLR-Stage2-7B-converted', 'steps', 12)}[a.model]
    model = Qwen2VLChatUniVLR(model_path=R + '/Models/' + cfg[0], decoding_strategy=cfg[1], univlr_steps=cfg[2],
                              clean_univlr_output=True, max_new_tokens=512, max_pixels=4014080)
    ds = build_dataset(a.ds); model.set_dump_image(ds.dump_image)
    rows = [r for _, r in ds.data.iterrows()][a.shard::a.nshard]
    if a.limit:
        rows = rows[:a.limit]
    fo = open(fp, 'w', buffering=1)
    for line in rows:
        msg = model.build_prompt(line, dataset=a.ds) if model.use_custom_prompt(a.ds) else ds.build_prompt(line)
        ch = {c: str(line[c]) for c in string.ascii_uppercase[:6] if c in line and not (isinstance(line[c], float) and math.isnan(line[c]))}
        gold = str(line['answer']).strip()

        def pred(out):
            p = can_infer(out, ch) if ch else None
            if not p:
                mm = re.search(r'\b([A-F])\b', out); p = mm.group(1) if mm else 'Z'
            return str(p)
        res = {}
        for md in ('head', 'zero', 'noise', 'swap'):
            ST.update(mode=md, n=0, i=0)
            if md == 'head':
                ST['cur'] = []
            out = model.generate(message=msg, dataset=a.ds); p = pred(out)
            res[md] = dict(pred=p, ok=p == gold[:1], out=out[:40], n_feed=ST['n'])
        ST['prev'] = ST['cur']
        fo.write(json.dumps(dict(key=int(line['index']), gold=gold, res=res), ensure_ascii=False) + '\n')
        print('G0', a.ds, int(line['index']), {md: (v['pred'], v['n_feed']) for md, v in res.items()}, flush=True)
    print('G0_DONE', a.ds, a.shard, len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ds', default='HRBench4K')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--model', choices=['s1', 's2'], default='s1')
    main(p.parse_args())
