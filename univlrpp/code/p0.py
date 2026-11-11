"""方向B P0 (2026-11-11, dsw3): slot-interface Visual-CoT SFT from Qwen2.5-VL-7B-Instruct.
Sequence = chat template [system][user: image + question][assistant: <|univlr_start|> K x <|univlr|> <|univlr_end|>
<|univlr_latent_end|> answer <|im_end|>]. Slots are injected at the input embeddings with Z = merged ViT features of an
image resized to a K-token grid: Z_base = ViT(problem image) (deployment) or Z_canvas = ViT(bbox crop) (privileged, with
probability --pcanvas). Strict slot-only mask with probability pmask (annealed 0 -> --pmask over the first --anneal of
the updates): every row after the image except the slots is blocked from the image tokens, so the answer sees the image
only through the slots. Loss = CE on the answer tokens. Full LLM (ViT + merger frozen), DeepSpeed ZeRO-2, image budget
max_pixels 4,014,080 = 5120 visual tokens (official Stage1 recipe).
phases: build | smoke (selftest + short training + holdout CE by slot condition) | train
"""
import os, sys, json, re, time, math, random, argparse, collections, datetime
from pathlib import Path
import torch

R = Path('/mnt/cpfs/jieduan/univlr')
QWEN = R / 'Models/Qwen2.5-VL-7B-Instruct'
MON = R / 'Datasets/Monet-SFT-125K'
P0 = R / 'experiments/p0_20261111'
TOKS = ['<|univlr_start|>', '<|univlr|>', '<|univlr_latent_end|>', '<|univlr_end|>']  # ids 151665..151668 (Stage1 names)
START, SLOT, LEND, END = 151665, 151666, 151667, 151668
MAXP, MINP = 5120 * 28 * 28, 128 * 28 * 28


def span(k):
    return '<|univlr_start|>' + '<|univlr|>' * k + '<|univlr_end|><|univlr_latent_end|>'


def clean_answer(t):
    t = t.replace('<abs_vis_token></abs_vis_token>', '')
    return re.sub(r'\s+', ' ', re.sub(r'</?observation>', '', t)).strip()


# ---------------------------------------------------------------- build
def build(a):
    d = json.load(open(MON / 'Visual_CoT/train.json')); stat = collections.Counter(); ss = []
    for r in d:
        msgs = {m['role']: m for m in r['data']}
        u, s_ = msgs.get('user'), msgs.get('assistant')
        iu = [c['image'] for c in u['content'] if c.get('type') == 'image']
        ia = [c['image'] for c in s_['content'] if c.get('type') == 'image']
        q = ''.join(c['text'] for c in u['content'] if c.get('type') == 'text').strip()
        ans = clean_answer(''.join(c['text'] for c in s_['content'] if c.get('type') == 'text'))
        if len(iu) != 1 or len(ia) != 1:
            stat['bad_images'] += 1; continue
        if not q or not ans:
            stat['empty'] += 1; continue
        ss.append({'key': f"vcot:{r['metadata']['sample_id']}", 'img': iu[0], 'crop': ia[0], 'q': q, 'a': ans})
    random.Random(20261111).shuffle(ss)
    (P0 / 'data').mkdir(parents=True, exist_ok=True)
    ho, tr = ss[:a.nhold], ss[a.nhold:]
    json.dump(ho, open(P0 / 'data/holdout.json', 'w'), ensure_ascii=False)
    json.dump(tr, open(P0 / 'data/train.json', 'w'), ensure_ascii=False)
    L = sorted(len(s['a']) for s in ss)
    print('BUILD', len(ss), 'train', len(tr), 'holdout', len(ho), dict(stat), 'answer chars p50/p99/max',
          L[len(L) // 2], L[int(.99 * len(L))], L[-1], flush=True)
    for s in ss[:3]:
        print('EXAMPLE', s['key'], repr(s['q'][:80]), '->', repr(s['a'][:80]), flush=True)


# ---------------------------------------------------------------- processor / data
def processor():
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(str(QWEN), min_pixels=MINP, max_pixels=MAXP)
    p.tokenizer.add_tokens(TOKS, special_tokens=True)
    got = p.tokenizer.convert_tokens_to_ids(TOKS)
    assert got == [START, SLOT, LEND, END], got
    return p


def grid_for(k, w, h):
    best = None
    for r in range(1, k + 1):
        if k % r:
            continue
        c = k // r; err = abs(math.log((c / r) / (w / h)))
        if best is None or err < best[0]:
            best = (err, r, c)
    return best[1], best[2]


def zimage(proc, im, k):
    """image resized to a k-token merged grid (each merged token = 28x28 px) -> processor tensors."""
    from PIL import Image
    r, c = grid_for(k, *im.size)
    x = proc.image_processor(images=[im.resize((c * 28, r * 28), Image.BICUBIC)], return_tensors='pt')
    thw = x['image_grid_thw']; assert int(thw.prod()) // 4 == k, (thw.tolist(), k)
    return x['pixel_values'], thw


def encode(proc, s, k):
    from PIL import Image
    im = Image.open(MON / s['img']).convert('RGB'); cr = Image.open(MON / s['crop']).convert('RGB')
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': s['q']}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True) + span(k) + s['a'] + '<|im_end|>'
    x = proc(text=[text], images=[im], return_tensors='pt')
    ids = x['input_ids']; a0 = int((ids[0] == LEND).nonzero()[0, 0]) + 1
    assert int((ids[0] == SLOT).sum()) == k
    zb, tb = zimage(proc, im, k); zc, tc = zimage(proc, cr, k)
    return {'key': s['key'], 'ids': ids, 'pv': x['pixel_values'], 'thw': x['image_grid_thw'], 'a0': a0,
            'zb': zb, 'tb': tb, 'zc': zc, 'tc': tc}


class DS:
    def __init__(self, ss, k):
        self.ss, self.k, self.proc = ss, k, None

    def __len__(self):
        return len(self.ss)

    def __getitem__(self, i):
        if self.proc is None:
            self.proc = processor()
        for j in range(8):  # a broken sample is replaced by the next one, so every rank still gets an item
            try:
                return encode(self.proc, self.ss[(i + j) % len(self.ss)], self.k)
            except Exception as err:  # noqa: BLE001
                print('DATA_SKIP', self.ss[(i + j) % len(self.ss)]['key'], repr(err)[:120], flush=True)
        raise RuntimeError('8 consecutive broken samples')


def prefetch(ds, nthreads, depth=8):
    """ordered iterator over ds with up to `depth` items prepared ahead by `nthreads` threads (no subprocesses)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, nthreads)) as ex:
        fut = [ex.submit(ds.__getitem__, i) for i in range(min(depth, len(ds)))]
        for i in range(len(ds)):
            b = fut[i].result()
            if i + depth < len(ds):
                fut.append(ex.submit(ds.__getitem__, i + depth))
            fut[i] = None
            yield b


# ---------------------------------------------------------------- model pieces
def vis(m, pv, thw):
    v = m.model.visual
    x = v(pv.to(m.device, torch.bfloat16), grid_thw=thw.to(m.device))
    return x[0] if isinstance(x, tuple) else x


def causal4(ids):
    """explicit causal bool mask. Never pass attention_mask=None: transformers 4.54 then infers packed sequences from the
    non-consecutive M-RoPE text positions and builds a block-diagonal mask (selftest d_causal 8.4 on 11-11)."""
    n = ids.shape[1]
    return torch.ones(n, n, dtype=torch.bool, device=ids.device).tril()[None, None]


def mask4(ids, slots_see=True):
    """bool [1,1,n,n], True = may attend. Causal; rows after the image that are not slots cannot see image tokens."""
    n = ids.shape[1]; img = ids[0] == IMG; dev = ids.device
    m = torch.ones(n, n, dtype=torch.bool, device=dev).tril()
    rows = torch.zeros(n, dtype=torch.bool, device=dev); rows[int(img.nonzero().max()) + 1:] = True
    if slots_see:
        rows[ids[0] == SLOT] = False
    m[rows[:, None] & img[None, :]] = False
    return m[None, None]


def embed(m, b, z, img_noise=0.0):
    ids = b['ids'].to(m.device)
    with torch.no_grad():
        ie = vis(m, b['pv'], b['thw'])
    if img_noise:
        ie = ie + img_noise * torch.randn_like(ie)
    emb = m.model.get_input_embeddings()(ids).clone()
    ipos = (ids[0] == IMG).nonzero()[:, 0]; spos = (ids[0] == SLOT).nonzero()[:, 0]
    assert len(ipos) == ie.shape[0], (len(ipos), ie.shape)
    emb[0, ipos] = ie.to(emb.dtype)
    if z is not None:
        assert len(spos) == z.shape[0], (len(spos), z.shape)
        emb[0, spos] = z.to(emb.dtype)
    pos, _ = m.model.get_rope_index(ids, b['thw'].to(m.device), None)
    return ids, emb, pos


def zfeat(m, b, which, other=None):
    with torch.no_grad():
        if which == 'base':
            return vis(m, b['zb'], b['tb'])
        if which == 'canvas':
            return vis(m, b['zc'], b['tc'])
        if which == 'swap':
            return vis(m, other['zb'], other['tb'])
        if which == 'zero':
            return torch.zeros(int((b['ids'][0] == SLOT).sum()), m.model.get_input_embeddings().weight.shape[1],
                               device=m.device, dtype=torch.bfloat16)
    raise ValueError(which)


class UniVLRHead(torch.nn.Module):
    """official UniVLRHead ('simple'): LayerNorm(eps 1e-6) -> Linear -> GELU -> Linear, hidden -> hidden."""
    def __init__(self, hidden_size):
        super().__init__()
        self.ln_q = torch.nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.GELU(),
                                       torch.nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        return self.mlp(self.ln_q(x))


def answer_ce(m, b, z, masked, img_noise=0.0, slots_see=True, return_logits=False, align_target=None, align_layer=14):
    """CE on the answer. With align_target ([K, hidden] = ViT(auxiliary image) at the K grid): also the per-position
    alignment MSE of univlr_head(hidden state of decoder layer `align_layer` at slot i) vs target token i (official
    Stage1: layer 14, tokenwise MSE); returns (ce, mse, pred)."""
    ids, emb, pos = embed(m, b, z, img_noise)
    am = mask4(ids, slots_see) if masked else causal4(ids)
    out = m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=am, use_cache=False,
                                 output_hidden_states=align_target is not None)
    n, a0 = ids.shape[1], b['a0']
    lg = m.lm_head(out.last_hidden_state[0, a0 - 1:n - 1]).float(); y = ids[0, a0:n]
    ce = torch.nn.functional.cross_entropy(lg, y)
    if align_target is not None:
        hs = out.hidden_states[align_layer][0, ids[0] == SLOT]  # index 0 = embeddings, L = decoder layer L output
        pred = m.univlr_head(hs)
        mse = torch.nn.functional.mse_loss(pred.float(), align_target.float().detach(), reduction='none').mean(-1).mean()
        return ce, mse, pred
    return (ce, lg) if return_logits else ce


# ---------------------------------------------------------------- selftest (rank 0, one holdout sample)
@torch.no_grad()
def selftest(m, b):
    import torch.nn.functional as F
    from transformers import DynamicCache
    was = m.training; m.eval()
    ids = b['ids'].to(m.device); a0 = b['a0']; n = ids.shape[1]; img = ids[0] == IMG; slot = ids[0] == SLOT
    am = mask4(ids)[0, 0]; last_img = int(img.nonzero().max())
    ok_struct = (bool(am[a0:][:, img].sum() == 0) and bool(am[slot][:, img].all())
                 and bool(am[last_img + 1:][~slot[last_img + 1:]][:, img].sum() == 0))
    z = zfeat(m, b, 'base'); lp = lambda x: F.log_softmax(x, -1)
    # reference = HF's own forward (2D all-ones mask, its own image injection / M-RoPE); ours without slot injection
    ref = m(input_ids=ids, pixel_values=b['pv'].to(m.device, torch.bfloat16), image_grid_thw=b['thw'].to(m.device),
            attention_mask=torch.ones_like(ids), use_cache=False).logits[0, a0 - 1:n - 1].float()
    _, l_own = answer_ce(m, b, None, False, return_logits=True)
    # bf16 kernels differ: eager / sdpa / flash disagree with each other by the same amount (diag3 11-11), so this is
    # reported, not asserted. Metric = mean KL over answer positions + gold CE gap (full-vocab max is tail noise).
    yy = ids[0, a0:n]
    d_causal = float(F.kl_div(lp(l_own), lp(ref), log_target=True, reduction='batchmean'))
    same = round(float((ref.argmax(-1) == l_own.argmax(-1)).float().mean()), 3)
    ce_gap = abs(float(F.cross_entropy(ref, yy)) - float(F.cross_entropy(l_own, yy)))
    ids_, emb0, pos0 = embed(m, b, None)
    o_none = m.model.language_model(inputs_embeds=emb0, position_ids=pos0, attention_mask=None, use_cache=False)
    d_none = float(F.kl_div(lp(m.lm_head(o_none.last_hidden_state[0, a0 - 1:n - 1]).float()), lp(ref), log_target=True, reduction='batchmean'))
    e2 = emb0.clone(); e2[0, -1] += 5.0 * torch.randn_like(e2[0, -1]); c4 = causal4(ids)
    h1 = m.model.language_model(inputs_embeds=emb0, position_ids=pos0, attention_mask=c4, use_cache=False).last_hidden_state
    h2 = m.model.language_model(inputs_embeds=e2, position_ids=pos0, attention_mask=c4, use_cache=False).last_hidden_state
    d_future = float((h1[0, :-1].float() - h2[0, :-1].float()).abs().max())
    _, emb, pos = embed(m, b, z)
    _, l_def = answer_ce(m, b, z, False, return_logits=True)
    _, l_m = answer_ce(m, b, z, True, return_logits=True); d_mask = float((lp(l_def) - lp(l_m)).abs().max())
    # leak test: with the slots also blocked from the image, image noise must not move the answer logits at all
    _, l_l0 = answer_ce(m, b, z, True, slots_see=False, return_logits=True)
    _, l_l1 = answer_ce(m, b, z, True, img_noise=5.0, slots_see=False, return_logits=True)
    d_leak = float((l_l0 - l_l1).abs().max())
    _, l_n1 = answer_ce(m, b, z, True, img_noise=5.0, return_logits=True); d_noise_slots = float((l_m - l_n1).abs().max())
    # cached decode step with the strict row mask == full masked forward at the same position
    full = mask4(ids)
    cache = DynamicCache()
    o1 = m.model.language_model(inputs_embeds=emb[:, :a0], position_ids=pos[:, :, :a0], attention_mask=full[:, :, :a0, :a0],
                                past_key_values=cache, use_cache=True)
    o2 = m.model.language_model(inputs_embeds=emb[:, a0:a0 + 1], position_ids=pos[:, :, a0:a0 + 1],
                                attention_mask=full[:, :, a0:a0 + 1, :a0 + 1], past_key_values=o1.past_key_values, use_cache=True)
    l_step = m.lm_head(o2.last_hidden_state[0, -1]).float()
    d_step = float((lp(l_step) - lp(l_m[1])).abs().max()); step_same = int(l_step.argmax()) == int(l_m[1].argmax())
    rec = dict(key=b['key'], n=n, n_img=int(img.sum()), a0=a0, struct=ok_struct, kl_ref=round(d_causal, 4), ce_gap_ref=round(ce_gap, 4), argmax_agree_ref=same, kl_none_path=round(d_none, 4), d_future=d_future,
               d_mask=round(d_mask, 4), d_leak=d_leak, d_noise_slots=round(d_noise_slots, 4), d_step=round(d_step, 4), step_same=step_same)
    print('SELFTEST', rec, flush=True); m.train(was)
    assert ok_struct and d_future == 0.0 and d_leak < 1e-3 and rec['n_img'] <= 5120 and d_noise_slots > 0 and step_same and d_step < 1.0, rec
    return rec


# ---------------------------------------------------------------- holdout CE by slot condition
COND = [('base_m', 'base', True), ('canvas_m', 'canvas', True), ('swap_m', 'swap', True), ('zero_m', 'zero', True),
        ('base_u', 'base', False)]


@torch.no_grad()
def hold_eval(m, items, rank, world):
    import torch.distributed as dist
    al = hasattr(m, 'univlr_head'); nc = len(COND)
    was = m.training; m.eval(); tot = torch.zeros(nc + 4, dtype=torch.float64, device=m.device)
    for i, b in enumerate(items):
        other = items[(i + 1) % len(items)]
        for j, (_, src, masked) in enumerate(COND):
            tot[j] += float(answer_ce(m, b, zfeat(m, b, src, other), masked))
        if al:  # head(layer-L slot states) vs ViT(auxiliary image); baseline = the injected ViT(image) itself
            zb, zc = zfeat(m, b, 'base'), zfeat(m, b, 'canvas')
            _, mse, pred = answer_ce(m, b, zb, True, align_target=zc, align_layer=m.p0_align_layer)
            cos = torch.nn.functional.cosine_similarity
            tot[nc] += float(mse); tot[nc + 1] += float(cos(pred.float(), zc.float(), -1).mean())
            tot[nc + 2] += float(cos(zb.float(), zc.float(), -1).mean())
        tot[-1] += 1
    dist.all_reduce(tot); m.train(was)
    r = {c[0]: round(float(tot[j] / tot[-1]), 4) for j, c in enumerate(COND)} | {'n': int(tot[-1])}
    if al:
        r |= {'align_mse_m': round(float(tot[nc] / tot[-1]), 4), 'cos_pred_aux': round(float(tot[nc + 1] / tot[-1]), 4),
              'cos_image_aux': round(float(tot[nc + 2] / tot[-1]), 4)}
    return r


# ---------------------------------------------------------------- training
def run(a, smoke):
    global IMG
    import deepspeed
    import torch.distributed as dist
    from transformers import Qwen2_5_VLForConditionalGeneration
    deepspeed.init_distributed(dist_backend='nccl', timeout=datetime.timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(lr_)
    X = P0 / a.sub; (X / 'logs').mkdir(parents=True, exist_ok=True)
    proc = processor(); IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(QWEN), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                           device_map={'': lr_})
    m.config.use_cache = False
    with torch.no_grad():  # the 4 new token rows are untrained padding rows of Qwen-Instruct: start them at the mean
        for W_ in (m.model.get_input_embeddings().weight, m.lm_head.weight):
            W_[START:END + 1] = W_[:START].float().mean(0).to(W_.dtype)
    for p in m.model.visual.parameters():
        p.requires_grad_(False)
    if a.align_w > 0:  # P0b: the auxiliary image is the alignment target only, it never enters the slots
        assert a.pcanvas == 0, 'pcanvas must be 0 when the auxiliary image is the alignment target'
        torch.manual_seed(20261111)  # same head init on every rank
        m.univlr_head = UniVLRHead(m.model.get_input_embeddings().weight.shape[1]).to(m.device, torch.bfloat16)
        m.p0_align_layer = a.align_layer
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tr = json.load(open(P0 / 'data/train.json')); ho = json.load(open(P0 / 'data/holdout.json'))
    if a.max_train:
        tr = tr[:a.max_train]
    per = len(tr) // world; mine = tr[rank * per:(rank + 1) * per]
    total = max(1, per // a.accum)
    hold_items = [encode(proc, s, a.k) for s in ho[:a.nhold_eval][rank::world]]
    if rank == 0:
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print('TRAIN P0', 'smoke' if smoke else 'full', 'samples', per * world, 'world', world, 'accum', a.accum,
              'global_batch', world * a.accum, 'updates', total, 'lr', a.lr, 'k', a.k, 'pmask', a.pmask, 'anneal', a.anneal,
              'pcanvas', a.pcanvas, 'align_w', a.align_w, 'align_layer', a.align_layer, 'head_lr', a.head_lr,
              'trainable', n_tr, flush=True)
        selftest(m, hold_items[0])
    dist.barrier()
    params = [p for p in m.parameters() if p.requires_grad]
    hid = {id(p) for p in m.univlr_head.parameters()} if hasattr(m, 'univlr_head') else set()
    groups = [{'params': [p for p in params if id(p) not in hid]}]
    scales = [1.0]
    if hid:  # official Stage1: univlr_head_lr 1e-4 vs LLM 1e-5
        groups.append({'params': [p for p in params if id(p) in hid]}); scales.append(a.head_lr / a.lr)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.1, betas=(0.9, 0.999))
    cfg = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': a.accum, 'gradient_clipping': 1.0,
           'bf16': {'enabled': True}, 'zero_optimization': {'stage': 2, 'overlap_comm': False, 'contiguous_gradients': True,
                                                            'reduce_bucket_size': 2e8, 'allgather_bucket_size': 2e8},
           'zero_allow_untested_optimizer': True, 'steps_per_print': 10 ** 9, 'wall_clock_breakdown': False}
    eng, opt, _, _ = deepspeed.initialize(model=m, model_parameters=params, optimizer=opt, config=cfg)
    mm = eng.module
    log = open(X / 'logs/train.jsonl', 'a', buffering=1) if rank == 0 else None
    h0 = hold_eval(mm, hold_items, rank, world)
    if rank == 0:
        print('HOLD before', h0, flush=True); log.write(json.dumps({'hold_before': h0}) + '\n')
    # no DataLoader worker processes: on dsw3 both tensor-sharing paths fail (fd over a unix socket -> Errno 95,
    # file_system -> torch_shm_manager 'Operation not supported'), so samples are prepared by a thread pool instead
    loader = prefetch(DS(mine, a.k), a.workers)
    warm = max(1, int(0.05 * total)); rng = random.Random(1111 + rank)
    lr_at = lambda u: a.lr * min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(u, total) / total))
    pm_at = lambda u: a.pmask * min(1.0, u / max(1, a.anneal * total))
    mm.train(); hist = collections.defaultdict(list); upd = 0; t0 = time.time(); tu = time.time()
    for i, b in enumerate(loader):
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = lr_at(upd) * scales[gi]
        masked = rng.random() < pm_at(upd); src = 'canvas' if rng.random() < a.pcanvas else 'base'
        if a.align_w > 0:
            ce, mse, _ = answer_ce(mm, b, zfeat(mm, b, 'base'), masked, align_target=zfeat(mm, b, 'canvas'), align_layer=a.align_layer)
            loss = ce + a.align_w * mse; hist['align_mse'].append(mse.item())
        else:
            ce = answer_ce(mm, b, zfeat(mm, b, src), masked); loss = ce
        eng.backward(loss)
        boundary = eng.is_gradient_accumulation_boundary()
        eng.step()
        hist[('m_' if masked else 'u_') + src].append(ce.item())
        if boundary:
            upd += 1
            if not smoke and a.save_every and upd % a.save_every == 0 and upd < total:
                if rank == 0:  # ZeRO-2 keeps full bf16 params on every rank
                    mm.save_pretrained(str(X / 'ckpt_last'), safe_serialization=True); proc.save_pretrained(str(X / 'ckpt_last'))
                    log.write(json.dumps({'saved': 'ckpt_last', 'update': upd}) + '\n')
                dist.barrier()
            if rank == 0 and (upd % a.log_every == 0 or upd == total):
                dt = (time.time() - tu) / a.log_every; tu = time.time()
                rec = {'update': upd, 'sample': (i + 1) * world, **{(k if k == 'align_mse' else f'ce_{k}'): round(float(sum(v[-50:]) / len(v[-50:])), 4) for k, v in hist.items()},
                       'n_by_cond': {k: len(v) for k, v in hist.items()}, 'pmask': round(pm_at(upd), 3), 'lr': lr_at(upd),
                       'sec_per_update': round(dt, 2), 'max_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                       'eta_full_epoch_h': round(dt * (118061 // (world * a.accum)) / 3600, 2)}
                print('STEP', rec, flush=True); log.write(json.dumps(rec) + '\n')
        if upd >= total:
            break
    h1 = hold_eval(mm, hold_items, rank, world)
    if rank == 0:
        print('HOLD after', h1, flush=True); log.write(json.dumps({'hold_after': h1, 'sec': round(time.time() - t0)}) + '\n')
        if not smoke:
            mm.save_pretrained(str(X / 'ckpt'), safe_serialization=True); proc.save_pretrained(str(X / 'ckpt'))
        print('TRAIN_DONE P0', round(time.time() - t0), 'updates', upd, flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['build', 'smoke', 'train'])
    p.add_argument('--sub', default='smoke')
    p.add_argument('--k', type=int, default=384)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--pmask', type=float, default=1.0)
    p.add_argument('--anneal', type=float, default=0.1)
    p.add_argument('--pcanvas', type=float, default=0.5)
    p.add_argument('--max-train', type=int, default=0)
    p.add_argument('--nhold', type=int, default=500)
    p.add_argument('--nhold-eval', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=2)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--align-w', type=float, default=0.0)
    p.add_argument('--align-layer', type=int, default=14)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--local_rank', type=int, default=-1)
    a = p.parse_args()
    if a.phase == 'build':
        build(a)
    else:
        run(a, a.phase == 'smoke')
