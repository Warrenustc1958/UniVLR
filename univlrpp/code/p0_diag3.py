"""P0 diag 3 (single PPU): both mask paths are causal but differ. Remaining difference = GQA handling (no mask:
sdpa(enable_gqa=True); 4D mask: repeat_kv then sdpa). (1) GQA kernel vs fp32 grouped reference; (2) model logits under
eager attention (plain torch, fp32 softmax) = ground truth, vs sdpa 2D-ones / sdpa causal4 / flash_attention_2."""
import sys, json, gc
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import torch, torch.nn.functional as F

torch.manual_seed(0); dev = 'cuda'
L, H, KV, Dh = 653, 28, 4, 128
q = torch.randn(1, H, L, Dh, device=dev, dtype=torch.bfloat16)
k, v = (torch.randn(1, KV, L, Dh, device=dev, dtype=torch.bfloat16) for _ in range(2))
tri = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
rep = lambda x: x.repeat_interleave(H // KV, dim=1)  # = transformers repeat_kv: q head h uses kv head h // 7
kr, vr = rep(k), rep(v)
ref = torch.softmax(((q.float() @ kr.float().transpose(-1, -2)) / Dh ** 0.5).masked_fill(~tri, float('-inf')), -1) @ vr.float()
tiled = lambda x: x.repeat(1, H // KV, 1, 1)  # alternative grouping: q head h uses kv head h % 4
ref_t = torch.softmax(((q.float() @ tiled(k).float().transpose(-1, -2)) / Dh ** 0.5).masked_fill(~tri, float('-inf')), -1) @ tiled(v).float()
for nm, o in (('enable_gqa+is_causal', F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)),
              ('repeat_kv+is_causal', F.scaled_dot_product_attention(q, kr, vr, is_causal=True)),
              ('repeat_kv+bool_tril', F.scaled_dot_product_attention(q, kr, vr, attn_mask=tri))):
    print('GQA', nm, 'vs grouped(h//7) fp32', round(float((o.float() - ref).abs().max()), 4),
          'vs tiled(h%4) fp32', round(float((o.float() - ref_t).abs().max()), 4), flush=True)

import p0
from transformers import Qwen2_5_VLForConditionalGeneration
proc = p0.processor(); p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
b = p0.encode(proc, json.load(open(p0.P0 / 'data/holdout.json'))[0], 384)
lp = lambda x: F.log_softmax(x.float(), -1)
res = {}
for impl in ('eager', 'sdpa', 'flash_attention_2'):
    try:
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(p0.QWEN), torch_dtype=torch.bfloat16, attn_implementation=impl,
                                                               device_map={'': 0}).eval()
    except Exception as err:  # noqa: BLE001
        print('LOAD_FAIL', impl, repr(err)[:200], flush=True); continue
    with torch.no_grad():
        ids, emb, pos = p0.embed(m, b, None); n, a0 = ids.shape[1], b['a0']
        masks = {'2D_ones': torch.ones_like(ids)}
        if impl != 'flash_attention_2':
            masks['causal4'] = p0.causal4(ids)
        for mn, am in masks.items():
            h = m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=am, use_cache=False).last_hidden_state
            res[f'{impl}/{mn}'] = m.lm_head(h[0, a0 - 1:n - 1]).float().cpu()
    del m; gc.collect(); torch.cuda.empty_cache()
base = res.get('eager/2D_ones')
y = b['ids'][0, b['a0']:]
for k_, v_ in res.items():
    d = (round(float((lp(v_) - lp(base)).abs().max()), 4), bool((v_.argmax(-1) == base.argmax(-1)).all())) if base is not None else None
    ce = round(float(F.cross_entropy(v_, y)), 4)
    print('MODEL', k_, 'vs eager/2D_ones', d, 'answer CE', ce, flush=True)
