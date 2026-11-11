"""read-only V* accuracy of a P0 prediction pkl with VLMEvalKit's can_infer (no report / cache writes)."""
import sys, pickle, string, collections
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
import pandas as pd
from vlmeval.utils.matching_util import can_infer
d = pd.read_csv('/mnt/cpfs/jieduan/univlr/Datasets/LMUData/VStarBench.tsv', sep='\t')
pred = pickle.load(open(sys.argv[1], 'rb'))
if isinstance(pred, pd.DataFrame):  # run.py dumps the prediction table (index, ..., prediction)
    pred = {int(i): p for i, p in zip(pred['index'], pred['prediction'])}
print('PKL type', type(pred).__name__, 'n', len(pred), 'key0', repr(list(pred)[0]) if isinstance(pred, dict) else '')
hit = unp = 0; by = collections.defaultdict(lambda: [0, 0]); ex = []
for _, r in d.iterrows():
    opts = {c: r[c] for c in string.ascii_uppercase if c in r and not pd.isna(r[c])}
    p = pred.get(int(r['index']), '')
    if isinstance(p, dict): p = p.get('prediction', '')
    a = can_infer(str(p), opts)
    ok = a == r['answer']; hit += ok; unp += a in (False, 'Z', None)
    by[r['category']][0] += ok; by[r['category']][1] += 1
    if not ok and len(ex) < 6: ex.append((str(p)[:50], 'gold', r['answer'], 'parsed', a))
print('ACC n', len(d), 'acc %.2f' % (100 * hit / len(d)), 'hit', hit, 'unparsed', unp)
for k, v in by.items(): print('BY', k, '%d/%d = %.1f' % (v[0], v[1], 100 * v[0] / v[1]))
for e in ex: print('EX', e)
