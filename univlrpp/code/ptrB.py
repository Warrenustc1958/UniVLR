"""Stage B pointer warm-start (2026-11-12, dsw3) for UniVLR++_LatentPointer_Glimpse_Design_20261112.md.
Backbone Qwen2.5-VL-7B-Instruct, frozen. Pointer forward = [system, user: image, question] + one pointer slot (inputs_embeds =
learnable e_ptr, M-RoPE position = prefix last + 1). The slot runs layers 0..13 over the prefix cache; its layer-14 input h is read:
  log pi_j = log_softmax_j( log(mean_h softmax_j(q_h . k_hj / sqrt(d))) / tau + delta_j )
  native term = Qwen's own layer-14 attention, heads 8 11 7 15 (G3'' 11-12: 19.5x chance untrained); q from the slot, k from the
  frozen image tokens; delta_j = Wq LN(h) . Wk LN(v_j) / 16 with Wq zero-init, tau = 1 -> update 0 == training-free pointer.
trainable: e_ptr; slot-only LoRA r16 on q/k/v/o of layers 0..13 and q of layer 14 (active only in the slot forward, so image and
question KV are untouched); Wq, Wk, LNs, tau. Loss = CE(m*, pi), m* = GT-box area overlap per merged 28 px cell (normalised).
Deployment (design §1, layout [image, crop, note + question]): image prefill (28 layers) -> question + slot to layer 14 only ->
read pi -> cache.crop(end of image) -> append crop + note + question. Unit tests U5-U8 check this cut point.
Data: Monet Visual_CoT (bbox from template matching the crop in the original, score >= 0.9) + Zebra visual_search exact crops
(score >= 0.95), minus the P0 holdout (eval) and benchmark-image duplicates (32x32 binary hash vs V*, HR4K/8K, MME-RW-Lite).
phases: build | unit | train (torchrun) | evalptr --ckpt init|<path> | e2e --shard --nshard | report
"""
import os, sys, json, math, copy, time, random, argparse, collections, datetime
from pathlib import Path
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path('/mnt/cpfs/jieduan/univlr')
X = R / 'experiments/ptrB_20261112'
MON = R / 'Datasets/Monet-SFT-125K'
QWEN = R / 'Models/Qwen2.5-VL-7B-Instruct'
P0HOLD = R / 'experiments/p0_20261111/data/holdout.json'
PL, HEADS, RANK = 14, [8, 11, 7, 15], 16
MINP, MAXP = 4 * 28 * 28, 5120 * 28 * 28


# ---------------------------------------------------------------- data build
def h32(gray):
    a = np.asarray(gray.resize((32, 32)), np.int16)
    return (a > a.mean()).tobytes().hex()


def _match(job):
    import cv2
    from PIL import Image
    key, img, crop, q, src, thr = job
    try:
        o = Image.open(MON / img).convert('L'); c = Image.open(MON / crop).convert('L')
        W, H = o.size; w, h = c.size
        if w > W or h > H or w < 4 or h < 4:
            return None
        f = 0.5 if W * H > 3_000_000 else 1.0
        O = np.asarray(o.resize((int(W * f), int(H * f))) if f < 1 else o, np.float32)
        C = np.asarray(c.resize((max(2, int(w * f)), max(2, int(h * f)))) if f < 1 else c, np.float32)
        r = cv2.matchTemplate(O, C, cv2.TM_CCOEFF_NORMED); _, mx, _, loc = cv2.minMaxLoc(r)
        if mx < thr:
            return None
        x0, y0 = loc[0] / f, loc[1] / f
        return dict(key=key, img=img, q=q, src=src, box=[x0, y0, x0 + w, y0 + h], W=W, H=H, score=round(float(mx), 4), h=h32(o))
    except Exception:
        return None


def _bench_hash(p):
    from PIL import Image
    try:
        return h32(Image.open(p).convert('L'))
    except Exception:
        return None


def rows(sub, prefix, thr):
    out = []
    for r in json.load(open(MON / sub / 'train.json')):
        msgs = {m['role']: m for m in r['data']}
        u, s_ = msgs.get('user'), msgs.get('assistant')
        iu = [c['image'] for c in u['content'] if c.get('type') == 'image']
        ia = [c['image'] for c in s_['content'] if c.get('type') == 'image']
        q = ''.join(c['text'] for c in u['content'] if c.get('type') == 'text').strip()
        if len(iu) == 1 and ia and q:
            out.append((f"{prefix}:{r['metadata']['sample_id']}", iu[0], ia[-1], q, sub, thr))
    return out


def build(a):
    from multiprocessing import Pool
    from vlmeval.dataset import build_dataset
    (X / 'data').mkdir(parents=True, exist_ok=True); st = collections.Counter()
    jobs = rows('Visual_CoT', 'vcot', 0.9) + rows('Zebra_CoT_visual_search', 'zs', 0.95)
    st['rows_vcot'] = sum(j[4] == 'Visual_CoT' for j in jobs); st['rows_zs'] = len(jobs) - st['rows_vcot']
    paths = set()
    for dsn in ('VStarBench', 'HRBench4K', 'HRBench8K', 'MME-RealWorld-Lite'):
        ds = build_dataset(dsn)
        for _, line in ds.data.iterrows():
            paths.update(ds.dump_image(line))
    with Pool(48) as pool:
        bench = {h for h in pool.map(_bench_hash, sorted(paths), chunksize=16) if h}
        res = [r for r in pool.imap_unordered(_match, jobs, chunksize=64) if r]
    st['bench_images'] = len(paths); st['matched'] = len(res)
    hold = {s['key'] for s in json.load(open(P0HOLD))}
    tr, ho = [], []
    for r in res:
        if r['h'] in bench:
            st['dup_bench'] += 1; continue
        (ho if r['key'] in hold else tr).append(r)
    random.Random(20261112).shuffle(tr)
    st['train'] = len(tr); st['holdout'] = len(ho)
    st['train_by_src'] = dict(collections.Counter(r['src'] for r in tr))
    fr = sorted((r['box'][2] - r['box'][0]) * (r['box'][3] - r['box'][1]) / (r['W'] * r['H']) for r in tr)
    st['box_area_frac_p10_p50_p90'] = [round(fr[len(fr) // 10], 4), round(fr[len(fr) // 2], 4), round(fr[9 * len(fr) // 10], 4)]
    json.dump(tr, open(X / 'data/train.json', 'w')); json.dump(ho, open(X / 'data/holdout.json', 'w'))
    json.dump(dict(st), open(X / 'data/stats.json', 'w'), indent=1); print('BUILD', json.dumps(dict(st)), flush=True)


# ---------------------------------------------------------------- model / pointer
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(QWEN), min_pixels=MINP, max_pixels=MAXP)


def load_base(dev):
    import p0
    from transformers import Qwen2_5_VLForConditionalGeneration
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(QWEN), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': dev}).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    proc = processor(); p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    return m, proc


def prefix_text(proc, q):
    t = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': q}]}], add_generation_prompt=True)
    return t[:t.index('<|im_end|>', t.index('<|vision_end|>'))]


class Ptr(nn.Module):
    def __init__(self, m, r=RANK):
        super().__init__()
        cfg = m.model.language_model.config; D = cfg.hidden_size; KV = cfg.num_key_value_heads * (D // cfg.num_attention_heads)
        self.names = [(L, n) for L in range(PL) for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj')] + [(PL, 'q_proj')]
        dims = {'q_proj': (D, D), 'k_proj': (D, KV), 'v_proj': (D, KV), 'o_proj': (D, D)}
        self.A = nn.ParameterList(); self.B = nn.ParameterList()
        for _, n in self.names:
            i, o = dims[n]; A = torch.empty(r, i); nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.A.append(nn.Parameter(A)); self.B.append(nn.Parameter(torch.zeros(o, r)))
        self.scale = 32.0 / r
        self.e_ptr = nn.Parameter(m.model.get_input_embeddings().weight[:151643].float().mean(0).cpu())
        self.lnq, self.lnk = nn.LayerNorm(D), nn.LayerNorm(D)
        self.Wq, self.Wk = nn.Linear(D, 256, bias=False), nn.Linear(D, 256, bias=False)
        nn.init.zeros_(self.Wq.weight)
        self.log_tau = nn.Parameter(torch.zeros(()))

    def groups(self):
        lora = list(self.A) + list(self.B)
        ids = {id(p) for p in lora}
        return lora, [p for p in self.parameters() if id(p) not in ids]


class Stop(Exception):
    pass


class Engine:
    """hooks on the frozen backbone: layer-14 pre-hook (read + optionally stop) and slot-only LoRA (gated)."""

    def __init__(self, m, ptr):
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as MQ
        self.MQ = MQ; self.m, self.ptr = m, ptr; self.lm = m.model.language_model; cfg = self.lm.config
        self.NH, self.NKV = cfg.num_attention_heads, cfg.num_key_value_heads; self.HD = cfg.hidden_size // self.NH
        self.grp = self.NH // self.NKV
        self.MS = (getattr(self.lm.layers[0].self_attn, 'rope_scaling', None) or cfg.rope_scaling)['mrope_section']
        self.L14 = self.lm.layers[PL]; self.att = self.L14.self_attn; self.mode = None; self.lora_on = False; self.c = {}
        self.L14.register_forward_pre_hook(self._pre, with_kwargs=True)
        for i, (L, n) in enumerate(ptr.names):
            getattr(self.lm.layers[L].self_attn, n).register_forward_hook(self._lora(i))

    def _pre(self, mod, args, kwargs):
        if self.mode is None:
            return None
        h = args[0] if args else kwargs['hidden_states']; cos, sin = kwargs['position_embeddings']
        self.c.update(h=h, cos=cos, sin=sin)
        if self.mode == 'stop':
            raise Stop
        return None

    def _lora(self, i):
        def hk(md, inp, out):
            if not self.lora_on:
                return out
            P = self.ptr
            return out + (inp[0] @ P.A[i].to(out.dtype).t() @ P.B[i].to(out.dtype).t()) * P.scale
        return hk

    def run(self, mode, **kw):
        self.mode = mode
        try:
            return self.lm(**kw)
        except Stop:
            return None
        finally:
            self.mode = None

    def image_keys(self, h_img, cos, sin):
        x = self.L14.input_layernorm(h_img[None]); n = h_img.shape[0]
        k = self.att.k_proj(x).view(1, n, self.NKV, self.HD).transpose(1, 2)
        k = self.MQ.apply_multimodal_rotary_pos_emb(k, k, cos, sin, self.MS)[1]
        return k[0, [hh // self.grp for hh in HEADS]]

    @torch.no_grad()
    def prefix(self, b):
        """training / eval path: [system, image, question] in one forward, stopped at layer 14 (cache = layers 0..13)."""
        import p0
        from transformers import DynamicCache
        ids, emb, pos = p0.embed(self.m, b, None); cache = DynamicCache()
        self.run('stop', inputs_embeds=emb, position_ids=pos, attention_mask=p0.causal4(ids), past_key_values=cache, use_cache=True)
        img = (ids[0] == p0.IMG).nonzero()[:, 0]; h = self.c['h']
        h_img = h[0, img]; kh = self.image_keys(h_img, self.c['cos'][:, :, img], self.c['sin'][:, :, img])
        _, hp, wp = [int(v) for v in b['thw'][0]]
        return dict(n=ids.shape[1], pos=pos, cache=cache, h_img=h_img, kh=kh, grid=(hp // 2, wp // 2), wp=wp, hp=hp)

    def slot(self, st, copy_cache=False):
        cache = copy.deepcopy(st['cache']) if copy_cache else st['cache']; dev = st['pos'].device
        e = self.ptr.e_ptr.to(torch.bfloat16)[None, None]; pos_s = st['pos'][:, :, -1:] + 1
        mask = torch.ones(1, 1, 1, st['n'] + 1, dtype=torch.bool, device=dev)
        self.lora_on = True
        try:
            self.run('stop', inputs_embeds=e, position_ids=pos_s, attention_mask=mask, past_key_values=cache, use_cache=True)
            hs, cs, ss = self.c['h'], self.c['cos'], self.c['sin']
            q = self.att.q_proj(self.L14.input_layernorm(hs))
        finally:
            self.lora_on = False
        q = q.view(1, 1, self.NH, self.HD).transpose(1, 2); q = self.MQ.apply_multimodal_rotary_pos_emb(q, q, cs, ss, self.MS)[0]
        sn = torch.einsum('hd,hnd->hn', q[0, HEADS, 0].float(), st['kh'].float()) / math.sqrt(self.HD)
        pn = torch.softmax(sn, -1).mean(0); P = self.ptr
        delta = P.Wk(P.lnk(st['h_img'].float())) @ P.Wq(P.lnq(hs[0, 0].float())) / 16.0
        return torch.log_softmax(torch.log(pn + 1e-9) / P.log_tau.exp() + delta, -1)


def make_ptr(m, dev, seed=20261112):
    torch.manual_seed(seed); return Ptr(m).to(dev)


def target(box, W, H, hp, wp):
    gh, gw = hp // 2, wp // 2; sx, sy = wp * 14 / W, hp * 14 / H
    b = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    xs = torch.arange(gw, dtype=torch.float32) * 28; ys = torch.arange(gh, dtype=torch.float32) * 28
    ox = (torch.minimum(xs + 28, torch.tensor(b[2])) - torch.maximum(xs, torch.tensor(b[0]))).clamp(min=0)
    oy = (torch.minimum(ys + 28, torch.tensor(b[3])) - torch.maximum(ys, torch.tensor(b[1]))).clamp(min=0)
    m = oy[:, None] * ox[None, :]
    if float(m.sum()) <= 0:
        m = torch.zeros(gh, gw); m[min(gh - 1, int((b[1] + b[3]) / 56)), min(gw - 1, int((b[0] + b[2]) / 56))] = 1
    return (m / m.sum()).flatten()


class DSet:
    def __init__(self, ss, proc, text_key='q'):
        self.ss, self.proc, self.tk = ss, proc, text_key

    def __len__(self):
        return len(self.ss)

    def __getitem__(self, i):
        from PIL import Image
        for j in range(8):
            s = self.ss[(i + j) % len(self.ss)]
            try:
                im = Image.open(s['img'] if os.path.isabs(s['img']) else MON / s['img']).convert('RGB')
                x = self.proc(text=[prefix_text(self.proc, s[self.tk])], images=[im], return_tensors='pt')
                _, hp, wp = [int(v) for v in x['image_grid_thw'][0]]
                return dict(ids=x['input_ids'], pv=x['pixel_values'], thw=x['image_grid_thw'], key=s['key'],
                            tgt=target(s['box'], s['W'], s['H'], hp, wp) if s.get('box') is not None else None, W=s['W'], H=s['H'])
            except Exception as err:  # noqa: BLE001
                print('DATA_SKIP', s.get('key'), repr(err)[:120], flush=True)
        raise RuntimeError('8 consecutive broken samples')


def metrics(logp, tgt):
    p = logp.exp(); inside = tgt > 0
    ce = float(-(tgt * logp).sum()); H = float(-(tgt[inside] * tgt[inside].log()).sum())
    return dict(ce=ce, kl=ce - H, mass=float(p[inside].sum()), base=float(inside.float().mean()), peak_in=bool(inside[int(p.argmax())]))


# ---------------------------------------------------------------- train
def train(a):
    import p0
    import torch.distributed as dist
    torch.set_grad_enabled(True)  # vlmeval/vlm/__init__.py calls torch.set_grad_enabled(False) at import time
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size(); ldev = int(os.environ.get('LOCAL_RANK', rank)); torch.cuda.set_device(ldev)
    (X / 'run').mkdir(parents=True, exist_ok=True); (X / 'logs').mkdir(parents=True, exist_ok=True)
    m, proc = load_base(ldev); ptr = make_ptr(m, f'cuda:{ldev}')
    for p in ptr.parameters():
        dist.broadcast(p.data, 0)
    eng = Engine(m, ptr); lora, head = ptr.groups(); params = lora + head
    tr = json.load(open(X / 'data/train.json')); ho = json.load(open(X / 'data/holdout.json'))
    per = len(tr) // world; mine = tr[rank::world][:per]; total = per // a.accum
    opt = torch.optim.AdamW([{'params': lora, 'lr': a.lr}, {'params': head, 'lr': a.head_lr}], weight_decay=0.0)
    base_lr = [a.lr, a.head_lr]; warm = max(1, int(0.03 * total))
    lr_at = lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(u, total) / total))
    log = open(X / 'logs/train.jsonl', 'a', buffering=1) if rank == 0 else None

    def hold_eval():
        tot = torch.zeros(6, dtype=torch.float64, device=f'cuda:{ldev}'); ds = DSet(ho[rank::world], proc)
        for i in range(len(ds)):
            b = ds[i]
            with torch.no_grad():
                mt = metrics(eng.slot(eng.prefix(b)).float(), b['tgt'].to(f'cuda:{ldev}'))
            tot += torch.tensor([mt['ce'], mt['kl'], mt['mass'], mt['base'], float(mt['peak_in']), 1.0], dtype=torch.float64, device=tot.device)
        dist.all_reduce(tot); n = float(tot[5])
        return dict(n=int(n), ce=round(float(tot[0]) / n, 4), kl=round(float(tot[1]) / n, 4), mass=round(float(tot[2]) / n, 4),
                    base=round(float(tot[3]) / n, 4), peak_in=round(float(tot[4]) / n, 4))
    h0 = hold_eval()
    if rank == 0:
        print('TRAIN ptrB samples', per * world, 'world', world, 'accum', a.accum, 'global_batch', world * a.accum, 'updates', total,
              'lr', a.lr, 'head_lr', a.head_lr, 'trainable', sum(p.numel() for p in params), 'HOLD before', h0, flush=True)
        log.write(json.dumps({'hold_before': h0}) + '\n')
    loader = p0.prefetch(DSet(mine, proc), a.workers)
    hist = collections.defaultdict(list); upd = 0; t0 = tu = time.time()
    for i, b in enumerate(loader):
        st = eng.prefix(b); logp = eng.slot(st); tgt = b['tgt'].to(logp.device)
        loss = -(tgt * logp.float()).sum(); (loss / a.accum).backward()
        with torch.no_grad():
            mt = metrics(logp.detach().float(), tgt)
        for k_ in ('ce', 'kl', 'mass', 'peak_in'):
            hist[k_].append(float(mt[k_]))
        hist['ratio'].append(mt['mass'] / max(mt['base'], 1e-6))
        if (i + 1) % a.accum == 0:
            g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params])
            dist.all_reduce(g); g /= world; off = 0
            for p in params:
                n_ = p.numel(); p.grad = g[off:off + n_].view_as(p).clone(); off += n_
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for gi, gr in enumerate(opt.param_groups):
                gr['lr'] = base_lr[gi] * lr_at(upd)
            opt.step(); opt.zero_grad(set_to_none=True); upd += 1
            if rank == 0 and (upd % a.log_every == 0 or upd == total):
                dt = (time.time() - tu) / a.log_every; tu = time.time()
                rec = {'update': upd, **{k_: round(float(np.mean(v[-200:])), 4) for k_, v in hist.items()}, 'tau': round(float(ptr.log_tau.exp()), 4),
                       'lr': base_lr[0] * lr_at(upd), 'sec_per_update': round(dt, 2), 'max_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
                print('STEP', rec, flush=True); log.write(json.dumps(rec) + '\n')
            if rank == 0 and upd % a.save_every == 0:
                torch.save({'ptr': ptr.state_dict(), 'update': upd}, X / 'run/ptrB_last.pt')
            if upd >= total:
                break
    h1 = hold_eval()
    if rank == 0:
        torch.save({'ptr': ptr.state_dict(), 'update': upd, 'args': vars(a)}, X / 'run/ptrB.pt')
        print('HOLD after', h1, flush=True); log.write(json.dumps({'hold_after': h1, 'sec': round(time.time() - t0)}) + '\n')
        print('TRAIN_DONE ptrB', round(time.time() - t0), 'updates', upd, flush=True)
    dist.barrier(); dist.destroy_process_group()


# ---------------------------------------------------------------- pointer eval (holdout + V*), single card
def load_ptr(m, ckpt, dev):
    ptr = make_ptr(m, dev)
    if ckpt != 'init':
        ptr.load_state_dict(torch.load(ckpt, map_location=dev)['ptr'])
    return ptr


def pointer_crop(logp, st, W, H):
    import g123 as G
    import gates2 as G2
    gh, gw = st['grid']; P = logp.exp().float().view(gh, gw).cpu()
    b1 = G2.regions(P, st['wp'] * 14 / W, st['hp'] * 14 / H)[0]
    return G.ctx_box(b1, W, H)


def evalptr(a):
    import g123 as G
    import gates2 as G2
    m, proc = load_base(0); ptr = load_ptr(m, a.ckpt, 'cuda:0'); eng = Engine(m, ptr)
    tag = 'init' if a.ckpt == 'init' else Path(a.ckpt).stem
    (X / 'eval').mkdir(parents=True, exist_ok=True); fo = open(X / f'eval/evalptr_{tag}.jsonl', 'w', buffering=1)
    ho = json.load(open(X / 'data/holdout.json'))
    vs = [dict(key=f"vstar:{it['key']}", img=it['img'], q=it['text'], box=it['box'], W=it['W'], H=it['H'], cat=it['cat']) for it in G.vstar_items()]
    for setn, ss in (('holdout', ho), ('vstar', vs)):
        ds = DSet(ss, proc)
        for i in range(len(ds)):
            b = ds[i]; s = ss[i]
            with torch.no_grad():
                st = eng.prefix(b); logp = eng.slot(st).float()
            mt = metrics(logp, b['tgt'].to(logp.device)); crop = pointer_crop(logp, st, s['W'], s['H'])
            gtc = G.ctx_box(s['box'], s['W'], s['H'])
            fo.write(json.dumps(dict(set=setn, key=s['key'], cat=s.get('cat', s.get('src')), **mt,
                                     contains=G2.contains(crop, s['box']), iou=round(G.iou(crop, gtc), 4))) + '\n')
    print('EVALPTR_DONE', tag, flush=True)


# ---------------------------------------------------------------- end to end (Qwen answers, old layout [image, crop, note + question])
def e2e(a):
    import g123 as G
    import g4hr
    from PIL import Image
    from vlmeval.utils.matching_util import can_infer
    m, proc = load_base(0); dev = 'cuda:0'
    pi = load_ptr(m, 'init', dev); pb = load_ptr(m, str(X / 'run/ptrB.pt'), dev); eng = Engine(m, pi)
    its = [dict(ds='vstar', **it) for it in G.vstar_items()] + [dict(ds='hr4k', **it) for it in g4hr.items()]
    its = its[a.shard::a.nshard]; (X / 'eval').mkdir(parents=True, exist_ok=True)
    fo = open(X / f'eval/e2e_shard{a.shard}.jsonl', 'w', buffering=1)
    for it in its:
        im = Image.open(it['img']).convert('RGB'); W, H = it['W'], it['H']
        x = proc(text=[prefix_text(proc, it['text'])], images=[im], return_tensors='pt')
        b = dict(ids=x['input_ids'], pv=x['pixel_values'], thw=x['image_grid_thw'])
        crops = {}
        for nm, P in (('ptr_init', pi), ('ptr_B', pb)):
            eng.ptr = P
            with torch.no_grad():
                st = eng.prefix(b); crops[nm] = pointer_crop(eng.slot(st).float(), st, W, H)
        if it['ds'] == 'vstar':
            crops['gt'] = G.ctx_box(it['box'], W, H)
        letters = sorted(it['choices']); gi = letters.index(it['gold']); res = {}
        conds = [('none', [im], it['text'])] + [(c, [im, G.render(im, crops[c])], G.NOTE + it['text']) for c in crops]
        for c, ims, txt in conds:
            out, lp, _ = G.ask(m, proc, ims, txt, letters)
            r = dict(ok=can_infer(out, it['choices']) == it['gold'], p_gold=round(float(torch.softmax(torch.tensor(lp), 0)[gi]), 4))
            if it['ds'] == 'vstar' and c in crops:
                import gates2 as G2
                r.update(contains=G2.contains(crops[c], it['box']), iou=round(G.iou(crops[c], G.ctx_box(it['box'], W, H)), 4))
            res[c] = r
        fo.write(json.dumps(dict(ds=it['ds'], key=it['key'], cat=it['cat'], res=res)) + '\n')
    print('E2E_DONE', a.shard, len(its), flush=True)


# ---------------------------------------------------------------- unit tests for the new cut point (design §2.7, U5-U8)
def unit(a):
    import p0
    import g123 as G
    from PIL import Image
    from transformers import DynamicCache
    m, proc = load_base(0); dev = 'cuda:0'; lm = m.model.language_model
    ptr = make_ptr(m, dev)
    with torch.no_grad():  # perturb so that LoRA and the delta term are active in the tests
        for B in ptr.B:
            B.normal_(0, 0.02)
        ptr.Wq.weight.normal_(0, 0.02); ptr.log_tau.fill_(0.1)
    eng = Engine(m, ptr); lp = lambda x: F.log_softmax(x.float(), -1); U = {}
    it = G.vstar_items()[0]; im = Image.open(it['img']).convert('RGB'); W, H = it['W'], it['H']
    crop = G.render(im, G.ctx_box(it['box'], W, H)); Q = it['text']
    t_full = G.chat(proc, [im, crop], G.NOTE + Q); xf = proc(text=[t_full + it['gold'] + '<|im_end|>'], images=[im, crop], return_tensors='pt')
    a0 = proc(text=[t_full], images=[im, crop], return_tensors='pt')['input_ids'].shape[1]
    xp = proc(text=[prefix_text(proc, Q)], images=[im], return_tensors='pt')
    idf, idp = xf['input_ids'], xp['input_ids']; ve = proc.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    c0 = int((idf[0] == ve).nonzero()[0, 0]) + 1; nf, npre = idf.shape[1], idp.shape[1]
    U['U5_prefix_tokens_equal'] = bool(torch.equal(idf[0, :c0], idp[0, :c0]))
    bf = dict(ids=idf, pv=xf['pixel_values'], thw=xf['image_grid_thw']); bp = dict(ids=idp, pv=xp['pixel_values'], thw=xp['image_grid_thw'])
    with torch.no_grad():
        _, ef, pf = p0.embed(m, bf, None); _, ep, pp = p0.embed(m, bp, None)
        U['U5_prefix_rope_equal'] = bool(torch.equal(pf[:, :, :c0], pp[:, :, :c0]))
        cf, cp = p0.causal4(idf.to(dev)), p0.causal4(idp.to(dev))
        ref = m.lm_head(lm(inputs_embeds=ef, position_ids=pf, attention_mask=cf, use_cache=False).last_hidden_state[0, a0 - 1:nf - 1]).float()
        # deployment: image prefill (28 layers, peek layer-14 image hidden) -> question to layer 14 -> slot -> crop cache -> append
        cache = DynamicCache(); eng.run('peek', inputs_embeds=ef[:, :c0], position_ids=pf[:, :, :c0], attention_mask=cf[:, :, :c0, :c0],
                                        past_key_values=cache, use_cache=True)
        img = (idf[0, :c0] == p0.IMG).nonzero()[:, 0]
        h_img = eng.c['h'][0, img]; kh = eng.image_keys(h_img, eng.c['cos'][:, :, img], eng.c['sin'][:, :, img])
        cache_np = copy.deepcopy(cache)
        eng.run('stop', inputs_embeds=ep[:, c0:], position_ids=pp[:, :, c0:], attention_mask=cp[:, :, c0:npre, :npre],
                past_key_values=cache, use_cache=True)
        _, hp, wp = [int(v) for v in bp['thw'][0]]
        st_dep = dict(n=npre, pos=pp, cache=cache, h_img=h_img, kh=kh, grid=(hp // 2, wp // 2), wp=wp, hp=hp)
        logp_dep = eng.slot(st_dep).float()
        cache.crop(c0)
        U['U5_cache_len_after_crop'] = [int(cache.layers[0].keys.shape[-2]), int(cache.layers[27].keys.shape[-2])] if hasattr(cache, 'layers') else None
        o = lm(inputs_embeds=ef[:, c0:], position_ids=pf[:, :, c0:], attention_mask=cf[:, :, c0:nf, :nf], past_key_values=cache, use_cache=True)
        dep = m.lm_head(o.last_hidden_state[0, a0 - 1 - c0:nf - 1 - c0]).float()
        o2 = lm(inputs_embeds=ef[:, c0:], position_ids=pf[:, :, c0:], attention_mask=cf[:, :, c0:nf, :nf], past_key_values=cache_np, use_cache=True)
        nop = m.lm_head(o2.last_hidden_state[0, a0 - 1 - c0:nf - 1 - c0]).float()
        U['U5_kl_deploy_vs_full'] = float(F.kl_div(lp(dep), lp(ref), log_target=True, reduction='batchmean'))
        U['U5_argmax_agree'] = float((dep.argmax(-1) == ref.argmax(-1)).float().mean())
        U['U5_maxdiff_deploy_vs_nopointer'] = float((dep - nop).abs().max())
        # U6: pointer readout, training path (one prefix forward to layer 14) vs deployment path
        logp_tr = eng.slot(eng.prefix(bp)).float()
        U['U6_kl_train_vs_deploy'] = float(F.kl_div(logp_dep, logp_tr, log_target=True, reduction='sum'))
        U['U6_maxdiff_prob'] = float((logp_tr.exp() - logp_dep.exp()).abs().max())
        U['U6_argmax_same'] = bool(int(logp_tr.argmax()) == int(logp_dep.argmax()))
    # U7: gradients reach only the pointer parameters (vlmeval import above disabled grad globally)
    torch.set_grad_enabled(True)
    st = eng.prefix(bp); n0 = st['cache'].layers[0].keys.shape[-2] if hasattr(st['cache'], 'layers') else None
    k_before = [st['cache'].layers[L].keys[..., :st['n'], :].clone() for L in range(PL)] if hasattr(st['cache'], 'layers') else []
    logp = eng.slot(st); tgt = target(it['box'], W, H, st['hp'], st['wp']).to(dev); (-(tgt * logp).sum()).backward()
    U['U7_base_params_with_grad'] = sum(p.grad is not None for p in m.parameters())
    U['U7_ptr_params_without_grad'] = sum(p.grad is None for p in ptr.parameters())
    U['U7_e_ptr_grad_norm'] = float(ptr.e_ptr.grad.norm())
    # U8: the slot forward (LoRA on) leaves the prefix KV untouched
    U['U8_prefix_kv_maxdiff'] = max(float((st['cache'].layers[L].keys[..., :st['n'], :] - k_before[L]).abs().max()) for L in range(PL)) if k_before else None
    U['U8_slot_appended'] = [int(n0), int(st['cache'].layers[0].keys.shape[-2])] if n0 is not None else None
    U['PASS'] = dict(U5=U['U5_prefix_tokens_equal'] and U['U5_prefix_rope_equal'] and U['U5_kl_deploy_vs_full'] < 1e-3 and U['U5_maxdiff_deploy_vs_nopointer'] == 0.0,
                     U6=U['U6_kl_train_vs_deploy'] < 5e-3 and U['U6_argmax_same'],
                     U7=U['U7_base_params_with_grad'] == 0 and U['U7_ptr_params_without_grad'] == 0 and U['U7_e_ptr_grad_norm'] > 0,
                     U8=U['U8_prefix_kv_maxdiff'] == 0.0)
    (X / 'eval').mkdir(parents=True, exist_ok=True); json.dump(U, open(X / 'eval/unit.json', 'w'), indent=1)
    print('UNIT', json.dumps(U), flush=True)
    assert all(U['PASS'].values()), U['PASS']
    print('UNIT_DONE', flush=True)


# ---------------------------------------------------------------- report
def report(a):
    import g123 as G
    L = ['# Stage B pointer warm-start (ptrB_20261112)', '']
    if (X / 'data/stats.json').exists():
        L += ['## data', '', '```', open(X / 'data/stats.json').read(), '```', '']
    if (X / 'eval/unit.json').exists():
        L += ['## unit tests U5-U8 (new cut point)', '', '```', open(X / 'eval/unit.json').read(), '```', '']
    if (X / 'logs/train.jsonl').exists():
        tl = [json.loads(l) for l in open(X / 'logs/train.jsonl')]
        L += ['## training', ''] + [f'- {json.dumps(r)}' for r in tl if 'hold_before' in r or 'hold_after' in r]
        st = [r for r in tl if 'update' in r and 'ce' in r]
        L += [f"- update {r['update']}: ce {r['ce']} kl {r['kl']} mass {r['mass']} ratio {r['ratio']} peak_in {r['peak_in']} tau {r['tau']}"
              for r in st[::max(1, len(st) // 10)] + st[-1:]] + ['']
    ev = {Path(f).stem.replace('evalptr_', ''): [json.loads(l) for l in open(f)] for f in sorted((X / 'eval').glob('evalptr_*.jsonl'))}
    if ev:
        L += ['## pointer quality (single peak region, expanded x2 = the crop rule)', '',
              '| ckpt | set | n | mass in GT / uniform | peak in GT | crop ⊇ GT box | IoU(crop, GT crop) | KL(m*‖π) |', '|:--|:--|--:|--:|--:|--:|--:|--:|']
        for tag, rs in ev.items():
            for setn in ('holdout', 'vstar'):
                r = [x for x in rs if x['set'] == setn]
                if r:
                    L.append(f"| {tag} | {setn} | {len(r)} | {np.mean([x['mass'] / max(x['base'], 1e-6) for x in r]):.1f} | "
                             f"{100 * np.mean([x['peak_in'] for x in r]):.1f}% | {100 * np.mean([x['contains'] for x in r]):.1f}% | "
                             f"{np.mean([x['iou'] for x in r]):.2f} | {np.mean([x['kl'] for x in r]):.3f} |")
        L.append('')
    rs = [json.loads(l) for f in sorted((X / 'eval').glob('e2e_shard*.jsonl')) for l in open(f)]
    for dsn in ('vstar', 'hr4k'):
        r = [x for x in rs if x['ds'] == dsn]
        if not r:
            continue
        conds = [c for c in ('none', 'ptr_init', 'ptr_B', 'gt') if c in r[0]['res']]
        ok = lambda x, c: int(x['res'][c]['ok'])
        L += [f'## end to end: {dsn} (n={len(r)}; answerer Qwen2.5-VL-7B-Instruct, layout [image, crop, note + question]; per-row acc)', '',
              '| condition | acc | Δ vs none [95% CI] | Δ vs ptr_init [95% CI] |', '|:--|--:|:--|:--|']
        for c in conds:
            d1 = G.boot([ok(x, c) - ok(x, 'none') for x in r]); d2 = G.boot([ok(x, c) - ok(x, 'ptr_init') for x in r])
            L.append(f"| {c} | {100 * np.mean([ok(x, c) for x in r]):.1f} | {100 * d1[0]:+.1f} [{100 * d1[1]:+.1f}, {100 * d1[2]:+.1f}] | "
                     f"{100 * d2[0]:+.1f} [{100 * d2[1]:+.1f}, {100 * d2[2]:+.1f}] |")
        cats = sorted({x['cat'] for x in r})
        L += ['', '| category | n | ' + ' | '.join(conds) + ' |', '|:--|--:|' + '--:|' * len(conds)]
        for cat in cats:
            rr = [x for x in r if x['cat'] == cat]
            L.append(f'| {cat} | {len(rr)} | ' + ' | '.join(f'{100 * np.mean([ok(x, c) for x in rr]):.1f}' for c in conds) + ' |')
        L.append('')
    (X / 'report.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['build', 'unit', 'train', 'evalptr', 'e2e', 'report'])
    p.add_argument('--ckpt', default='init')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--nshard', type=int, default=1)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--local_rank', type=int, default=-1)
    a = p.parse_args()
    {'build': build, 'unit': unit, 'train': train, 'evalptr': evalptr, 'e2e': e2e, 'report': report}[a.phase](a)
