"""P0 diag 2 (single PPU): 2D all-ones mask and an explicit 4D causal mask give different Qwen2.5-VL logits on PPU.
Which one is truly causal? (1) raw torch SDPA kernels vs an fp32 reference; (2) model-level causality: perturb the last
input embedding and measure the change at earlier positions (must be 0 for a causal model)."""
import sys, json, inspect
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import torch, torch.nn.functional as F

torch.manual_seed(0); dev = 'cuda'
print('BACKENDS flash', torch.backends.cuda.flash_sdp_enabled(), 'mem_eff', torch.backends.cuda.mem_efficient_sdp_enabled(),
      'math', torch.backends.cuda.math_sdp_enabled(), flush=True)
for L in (64, 653):
    q, k, v = (torch.randn(1, 28, L, 128, device=dev, dtype=torch.bfloat16) for _ in range(3))
    tri = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
    s = (q.float() @ k.float().transpose(-1, -2)) / 128 ** 0.5
    ref = torch.softmax(s.masked_fill(~tri, float('-inf')), -1) @ v.float()
    full = torch.softmax(s, -1) @ v.float()
    outs = {'is_causal': F.scaled_dot_product_attention(q, k, v, is_causal=True),
            'bool_tril': F.scaled_dot_product_attention(q, k, v, attn_mask=tri),
            'float_tril': F.scaled_dot_product_attention(q, k, v, attn_mask=torch.zeros(L, L, device=dev, dtype=torch.bfloat16).masked_fill(~tri, float('-inf'))),
            'no_mask': F.scaled_dot_product_attention(q, k, v)}
    for nm, o in outs.items():
        print('KERNEL L', L, nm, 'vs causal_fp32', round(float((o.float() - ref).abs().max()), 4),
              'vs noncausal_fp32', round(float((o.float() - full).abs().max()), 4), flush=True)

import p0
from transformers import Qwen2_5_VLForConditionalGeneration
proc = p0.processor(); p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(p0.QWEN), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                       device_map={'': 0}).eval()
b = p0.encode(proc, json.load(open(p0.P0 / 'data/holdout.json'))[0], 384)
with torch.no_grad():
    ids, emb, pos = p0.embed(m, b, None); n = ids.shape[1]
    emb2 = emb.clone(); emb2[0, -1] += 5.0 * torch.randn_like(emb2[0, -1])
    for nm, am in (('2D_ones', torch.ones_like(ids)), ('causal4', p0.causal4(ids))):
        h1 = m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=am, use_cache=False).last_hidden_state
        h2 = m.model.language_model(inputs_embeds=emb2, position_ids=pos, attention_mask=am, use_cache=False).last_hidden_state
        print('CAUSALITY', nm, 'max change at positions < n-1 after perturbing the last token:',
              round(float((h1[0, :-1].float() - h2[0, :-1].float()).abs().max()), 4), flush=True)
    # first-token check against greedy generation from the prompt (HF generate uses its own mask path)
from transformers.integrations import sdpa_attention
src = inspect.getsource(sdpa_attention.sdpa_attention_forward)
for l in src.splitlines():
    if 'is_causal' in l or 'attention_mask' in l:
        print('SRC', l.strip())
