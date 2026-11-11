"""read-only: paired ptr_B vs tbox / ptr_p0b on V* and HR4K (same items, same answerer and layout)."""
import json, glob, numpy as np
XP = '/mnt/cpfs/jieduan/univlr/experiments/ptrB_20261112/eval'; TB = '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/gates/tbox2'
e = {(r['ds'], r['key']): r for f in glob.glob(XP + '/e2e_shard*.jsonl') for r in map(json.loads, open(f))}
def boot(x):
    x = np.asarray(x, float); b = np.random.default_rng(0).choice(x, (4000, len(x))).mean(1)
    return 100 * x.mean(), 100 * np.quantile(b, .025), 100 * np.quantile(b, .975)
for ds in ('vstar', 'hr4k'):
    t = {r['key']: r for f in glob.glob(f'{TB}/{ds}_shard*.jsonl') for r in map(json.loads, open(f))}
    ks = [k for k in t if (ds, k) in e]
    B = lambda k: int(e[(ds, k)]['res']['ptr_B']['ok'])
    for o in ('tbox', 'ptr_p0b', 'none'):
        d = boot([B(k) - int(t[k]['res'][o]['ok']) for k in ks])
        print(ds, len(ks), f'ptr_B - {o}: {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]', 'acc', round(100 * np.mean([int(t[k]['res'][o]['ok']) for k in ks]), 1))
    print(ds, 'ptr_B acc', round(100 * np.mean([B(k) for k in ks]), 1))
