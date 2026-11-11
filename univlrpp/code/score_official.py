"""Official VLMEvalKit exact-matching scores from prediction pkl files (eval_20261110 + the 11-05 UniVLR-Stage2 run).
run.py's own scoring step fails in this env: the xlsx writer is missing, dump() falls back to pkl and evaluate() cannot
find the xlsx. Here each pkl is re-dumped as tsv with VLMEvalKit's dump() (PRED_FORMAT=tsv, an official prediction
format) and scored by the dataset object's own evaluate() with model='exact_matching' (what run.py --judge
exact_matching passes). Per-job summaries + per-item hits are cached in score/<row>/<dataset>.json (key = pkl size and
mtime); the report is rebuilt from every cached job, so runs on dsw3 / dsw4 only add rows.
usage: python score_official.py [--force] [--report-only]"""
import os, sys, json, glob, time, socket, hashlib, argparse, traceback, collections
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PRED_FORMAT'] = 'tsv'
R = '/mnt/cpfs/jieduan/univlr'
os.environ.setdefault('LMUData', f'{R}/Datasets/LMUData')
K = f'{R}/UniVLRplus/VLMEvalKit'; E = f'{R}/experiments/eval_20261110'; S = f'{E}/score'
sys.path.insert(0, K)
import numpy as np  # noqa: E402

O = f'{E}/outputs'
ROWS = [  # row, prediction dir, file prefix, description
    ('s1_noslots', f'{O}/s1_noslots/E20261110', 's1_noslots', 'Stage1 base (UniVLR-Stage1-7B), empty latent span'),
    ('s1_base', f'{O}/s1_base/E20261110', 's1_base', 'slots = ViT(image) only, no writer'),
    ('p1_writer', f'{O}/p1_writer/E20261110', 'p1_writer', 'P1 writer, frozen reader'),
    ('p2_writer', f'{O}/p2_writer/E20261110', 'p2_writer', 'P1 writer + P2 reader LoRA, image visible (reader was trained under the mask only)'),
    ('p2_base', f'{O}/p2_base/E20261110', 'p2_base', 'P2 reader LoRA, slots = ViT(image) only (no writer increment), image visible'),
    ('s1_noslots_m', f'{O}/s1_noslots_m/E20261110', 's1_noslots_m', 'masked, empty latent span = question only'),
    ('s1_base_m', f'{O}/s1_base_m/E20261110', 's1_base_m', 'masked, slots = ViT(image) only'),
    ('p1_writer_m', f'{O}/p1_writer_m/E20261110', 'p1_writer_m', 'masked, P1 writer, frozen reader'),
    ('p2_writer_m', f'{O}/p2_writer_m/E20261110', 'p2_writer_m', 'masked, P1 writer + P2 reader LoRA (the condition P2 was trained in)'),
    ('p2_base_m', f'{O}/p2_base_m/E20261110', 'p2_base_m', 'masked, P2 reader LoRA, slots = ViT(image) only (no writer increment)'),
    ('univlr_s2_1105', f'{R}/eval/UniVLR_Stage2_s12/T20261105-161253', 'UniVLR_Stage2_s12',
     'official UniVLR-Stage2-7B, predictions of 2026-11-05 (12 latent steps), rescored here'),
    ('s1_official', f'{R}/experiments/p0_20261111/step0/outputs/s1_official/E20261111', 's1_official',
     'official wrapper, UniVLR-Stage1-7B-converted (univlr, 1 step), max_pixels 4,014,080 = 5120 tokens'),
    ('qwen_instruct', f'{R}/experiments/p0_20261111/step0/outputs/qwen_instruct/E20261111', 'qwen_instruct',
     'official Qwen2VLChat, Qwen2.5-VL-7B-Instruct, max_pixels 4,014,080 = 5120 tokens'),
    ('p0_m', f'{R}/experiments/p0_20261111/eval/outputs/p0_m/E20261111', 'p0_m',
     'P0 (Qwen-Instruct + slot-interface Visual-CoT SFT), answer sees the image only through the 384 slots, 5120 tokens'),
    ('p0_u', f'{R}/experiments/p0_20261111/eval/outputs/p0_u/E20261111', 'p0_u',
     'P0, same checkpoint, plain causal attention (image visible to the answer), 5120 tokens'),
    ('p0s1500_m', f'{R}/experiments/p0_20261111/eval/outputs/p0s1500_m/E20261111', 'p0s1500_m',
     'P0 ckpt_last = update 1500 / 1844, answer sees the image only through the slots'),
    ('p0s1500_u', f'{R}/experiments/p0_20261111/eval/outputs/p0s1500_u/E20261111', 'p0s1500_u',
     'P0 ckpt_last = update 1500 / 1844, image visible to the answer'),
    ('p0f_m', f'{R}/experiments/p0_20261111/eval/outputs/p0f_m/E20261111', 'p0f_m', 'P0 final ckpt (1844), answer sees the image only through the slots'),
    ('p0f_u', f'{R}/experiments/p0_20261111/eval/outputs/p0f_u/E20261111', 'p0f_u', 'P0 final ckpt (1844), image visible to the answer'),
    ('p0b1000_u', f'{R}/experiments/p0_20261111/eval/outputs/p0b1000_u/E20261111', 'p0b1000_u',
     'P0b (+ layer-14 alignment to ViT(auxiliary image), pmask 0.5) ckpt_last = update 1000 / 1844, image visible'),
    ('p0bf_u', f'{R}/experiments/p0_20261111/eval/outputs/p0bf_u/E20261111', 'p0bf_u', 'P0b final ckpt (1844), image visible to the answer'),
    ('p0bf_m', f'{R}/experiments/p0_20261111/eval/outputs/p0bf_m/E20261111', 'p0bf_m', 'P0b final ckpt (1844), answer sees the image only through the slots'),
]
DS = ['VStarBench', 'HRBench4K', 'HRBench8K', 'MME-RealWorld-Lite']
PAPER = {'VStarBench': 82.7, 'HRBench4K': 73.3, 'HRBench8K': 68.8, 'MME-RealWorld-Lite': 50.7}
PAIRS = [('s1_base', 's1_noslots'), ('p1_writer', 's1_noslots'), ('p1_writer', 's1_base'), ('p2_writer', 'p1_writer'),
         ('p2_writer', 's1_noslots'), ('p2_writer', 'p2_base'), ('p2_base', 's1_noslots'),
         ('s1_base_m', 's1_noslots_m'), ('p1_writer_m', 's1_base_m'),
         ('p2_writer_m', 'p1_writer_m'), ('p2_writer_m', 's1_base_m'), ('p2_writer_m', 'p2_base_m'),
         ('p2_base_m', 's1_base_m'), ('p2_writer_m', 's1_noslots'), ('p2_base_m', 's1_noslots'),
         ('univlr_s2_1105', 's1_noslots'),
         ('s1_official', 's1_noslots'), ('s1_official', 'qwen_instruct'), ('univlr_s2_1105', 's1_official'),
         ('univlr_s2_1105', 'qwen_instruct'),
         ('p0_m', 's1_official'), ('p0_u', 's1_official'), ('p0_m', 'qwen_instruct'), ('p0_u', 'qwen_instruct'),
         ('p0_u', 'p0_m'),
         ('p0s1500_m', 's1_official'), ('p0s1500_u', 's1_official'), ('p0s1500_m', 'qwen_instruct'),
         ('p0s1500_u', 'p0s1500_m'),
         ('p0f_m', 's1_official'), ('p0f_u', 's1_official'), ('p0f_m', 'qwen_instruct'), ('p0f_u', 'qwen_instruct'),
         ('p0b1000_u', 'p0f_u'), ('p0b1000_u', 's1_official'), ('p0b1000_u', 'qwen_instruct'),
         ('p0bf_u', 's1_official'), ('p0bf_u', 'qwen_instruct'), ('p0bf_u', 'p0f_u'),
         ('p0bf_m', 'p0s1500_m'), ('p0bf_m', 'qwen_instruct'), ('p0bf_u', 'p0bf_m')]


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = f'{path}.tmp{socket.gethostname()}{os.getpid()}'
    json.dump(obj, open(t, 'w'), ensure_ascii=False); os.replace(t, path)


def pkl_key(pkl):
    st = os.stat(pkl); return [st.st_size, int(st.st_mtime)]


def cached(row, ds, pkl):
    c = f'{S}/{row}/{ds}.json'
    if os.path.exists(c):
        j = json.load(open(c))
        if j.get('key') == pkl_key(pkl):
            return j
    return None


def score_job(row, d, prefix, ds, dataset):
    from vlmeval.smp import load, dump
    pkl = f'{d}/{prefix}_{ds}.pkl'
    work = f'{S}/_work/{socket.gethostname()}_{os.getpid()}/{row}'; os.makedirs(work, exist_ok=True)
    for f in glob.glob(f'{work}/{prefix}_{ds}*'):
        os.remove(f)  # evaluate() reuses score files it finds
    data = load(pkl)
    info = {'key': pkl_key(pkl), 'row': row, 'ds': ds, 'pkl': pkl, 'dataset_class': type(dataset).__name__,
            'rows': len(data), 'uniq': int(data['index'].nunique()), 'expected': len(dataset.data),
            'empty': int(sum(str(x).strip() in ('', 'nan', 'None') for x in data['prediction']))}
    tsv = f'{work}/{prefix}_{ds}.tsv'; dump(data, tsv); back = load(tsv)
    info['roundtrip_diff'] = int(sum(str(x) != str(y) for x, y in zip(data['prediction'], back['prediction'])))
    res = dataset.evaluate(tsv, nproc=8, verbose=False, retry=6, timeout=600, model='exact_matching')
    info['raw'] = (json.dumps(res, ensure_ascii=False) if isinstance(res, dict) else res.to_string())[:6000]
    if ds.startswith('MME'):
        from vlmeval.dataset.utils.multiple_choice import extract_characters_regex
        sc = load(f'{work}/{prefix}_{ds}_score.tsv')
        hits = {str(i): float(v) for i, v in zip(sc['index'], sc['score'])}
        info['unparsed'] = int(sum(extract_characters_regex(str(x)) == '' for x in sc['prediction']))
        off = res['Overall']; info['extra'] = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    else:
        rf = load(f'{work}/{prefix}_{ds}_exact_matching_result.tsv')
        hits = {str(i): float(v) for i, v in zip(rf['index'], rf['hit'])}
        info['unparsed'] = int(sum('Failed in Prefetch' in str(x) for x in rf['log']))
        if 'Overall' in res.columns:
            off = float(res['Overall'].iloc[0]); info['extra'] = {}
        else:  # HRBench: cycle / type / accuracy, headline = Average over cycles of type all
            av = res[res['cycle'] == 'Average']; ex = {str(t): float(v) for t, v in zip(av['type'], av['accuracy'])}
            off = ex['all']; info['extra'] = ex
            opt = [c for c in 'ABCDEFGH' if c in data.columns]
            cid = lambda r: hashlib.sha1((str(r['question']) + '|' + '|'.join(sorted(str(r[c]) for c in opt))).encode()).hexdigest()[:12]
            info['cluster'] = {str(r['index']): cid(r) for _, r in data.iterrows()}
    mean_hit = 100 * float(np.mean(list(hits.values())))
    if isinstance(off, dict):
        off = off.get('Avg', off.get('acc', list(off.values())[0]))
    off = float(off); info['official_raw'] = off
    info['head'] = 100 * off if abs(100 * off - mean_hit) < abs(off - mean_hit) else off
    info['mean_hit'] = mean_hit; info['hits'] = hits
    info['scored_utc'] = time.strftime('%F %T', time.gmtime()); info['host'] = socket.gethostname()
    jdump(info, f'{S}/{row}/{ds}.json')
    return info


def paired(a, b, cl=None, B=10000, seed=0):
    idx = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in idx], float)
    g = collections.defaultdict(list)
    for j, i in enumerate(idx):
        g[cl.get(i, i) if cl else i].append(j)
    gs = np.array([d[v].sum() for v in g.values()]); gn = np.array([len(v) for v in g.values()], float)
    rng = np.random.default_rng(seed); n = len(gs); s = rng.integers(0, n, (B, n))
    bt = gs[s].sum(1) / gn[s].sum(1)
    return 100 * d.mean(), 100 * np.percentile(bt, 2.5), 100 * np.percentile(bt, 97.5), len(idx), n


def report():
    J = {}
    for row, _, _, _ in ROWS:
        for ds in DS:
            c = f'{S}/{row}/{ds}.json'
            if os.path.exists(c):
                J[row, ds] = json.load(open(c))
    f1 = lambda x: '%.1f' % x
    L = ['# Official VLMEvalKit protocol: dataset files, prompts and exact-matching scorer',
         '', f'Built {time.strftime("%F %T", time.gmtime())} UTC on {socket.gethostname()} by p1_20261110/code/score_official.py.',
         'Predictions come from run.py (--judge exact_matching); its scoring step failed (no xlsx writer), so each prediction',
         'pkl was re-dumped as tsv and scored with the dataset class\'s own evaluate(). Accuracy %, n = 191 / 800 / 800 / 1919.',
         'HRBench = Average over the cyclic option orders (type all).',
         '', '**Not the same input budget:** WriterVLM rows (s1_*, p1_*, p2_*) use the 方向B engine: one image, at most 1280',
         'visual tokens, answer scaffold with the latent span. univlr_s2_1105 used the official wrapper with the converted',
         'preprocessor default (max_pixels 4,014,080 = 5120 tokens). Rows ending in _m: strict slot-only mask (after the image',
         'only the slot positions may attend to image tokens).',
         '', '| row | what | ' + ' | '.join(DS) + ' |', '|:--|:--|' + '--:|' * len(DS)]
    for row, _, _, desc in ROWS:
        if any((row, ds) in J for ds in DS):
            L.append(f'| {row} | {desc} | ' + ' | '.join(f1(J[row, ds]['head']) if (row, ds) in J else '-' for ds in DS) + ' |')
    L.append('| paper UniVLR | reported in the paper, not rerun | ' + ' | '.join(f1(PAPER[ds]) for ds in DS) + ' |')
    L += ['', '## HRBench FSP / FCP (Average over cycles; single = FSP, cross = FCP)', '',
          '| row | HRBench4K FSP | HRBench4K FCP | HRBench8K FSP | HRBench8K FCP |', '|:--|--:|--:|--:|--:|']
    for row, _, _, _ in ROWS:
        if any((row, ds) in J for ds in ('HRBench4K', 'HRBench8K')):
            cell = []
            for ds in ('HRBench4K', 'HRBench8K'):
                ex = J[row, ds].get('extra', {}) if (row, ds) in J else {}
                sc = 100 if (row, ds) in J and J[row, ds]['official_raw'] <= 1 else 1
                cell += [f1(sc * ex[t]) if t in ex else '-' for t in ('single', 'cross')]
            L.append(f'| {row} | ' + ' | '.join(cell) + ' |')
    L += ['', '## Paired differences (percentage points, 95% bootstrap CI; HRBench resampled by question = its cyclic variants)', '',
          '| a − b | ' + ' | '.join(DS) + ' |', '|:--|' + ':--|' * len(DS)]
    for a, b in PAIRS:
        cell = []
        for ds in DS:
            if (a, ds) in J and (b, ds) in J:
                m, lo, hi, n, ng = paired(J[a, ds]['hits'], J[b, ds]['hits'], J[a, ds].get('cluster'))
                cell.append(f'{m:+.1f} [{lo:+.1f}, {hi:+.1f}]')
            else:
                cell.append('-')
        if any(c != '-' for c in cell):
            L.append(f'| {a} − {b} | ' + ' | '.join(cell) + ' |')
    L += ['', '## Checks', '', '| row | dataset | class | rows / unique / expected | empty | tsv round-trip diffs | unparsed by the scorer | official vs mean(hit) | scored (UTC, host) |',
          '|:--|:--|:--|:--|--:|--:|--:|:--|:--|']
    for row, _, _, _ in ROWS:
        for ds in DS:
            if (row, ds) in J:
                j = J[row, ds]
                L.append(f"| {row} | {ds} | {j['dataset_class']} | {j['rows']} / {j['uniq']} / {j['expected']} | {j['empty']} | {j['roundtrip_diff']} | "
                         f"{j['unparsed']} | {j['head']:.2f} vs {j['mean_hit']:.2f} | {j['scored_utc']} {j['host']} |")
    out = f'{E}/report.md'; t = f'{out}.tmp{socket.gethostname()}{os.getpid()}'
    open(t, 'w').write('\n'.join(L) + '\n'); os.replace(t, out); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); p.add_argument('--report-only', action='store_true')
    a = p.parse_args()
    if not a.report_only:
        from vlmeval.dataset import build_dataset
        for ds in DS:
            todo = [(row, d, prefix) for row, d, prefix, _ in ROWS
                    if os.path.exists(f'{d}/{prefix}_{ds}.pkl') and (a.force or cached(row, ds, f'{d}/{prefix}_{ds}.pkl') is None)]
            if not todo:
                continue
            dataset = build_dataset(ds)
            for row, d, prefix in todo:
                try:
                    j = score_job(row, d, prefix, ds, dataset)
                    print('SCORED', row, ds, '%.2f' % j['head'], 'mean_hit %.2f' % j['mean_hit'], 'unparsed', j['unparsed'], flush=True)
                except Exception:  # noqa: BLE001  one bad file must not hide the other rows
                    print('SCORE_FAIL', row, ds, traceback.format_exc()[-1500:], flush=True)
            del dataset
    report()
