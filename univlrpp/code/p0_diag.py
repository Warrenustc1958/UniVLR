"""P0 selftest failure diag (single card): where does our forward (embed + get_rope_index + language_model + causal4)
diverge from HF's own Qwen2.5-VL forward? One holdout sample, no slot injection (slot rows keep their token embedding)."""
import sys, json
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import torch, torch.nn.functional as F
import p0
from transformers import Qwen2_5_VLForConditionalGeneration

proc = p0.processor(); p0.IMG = proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(p0.QWEN), torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                       device_map={'': 0}).eval()
s = json.load(open(p0.P0 / 'data/holdout.json'))[0]
b = p0.encode(proc, s, 384)
ids = b['ids'].to(m.device); a0 = b['a0']; n = ids.shape[1]
pv = b['pv'].to(m.device, torch.bfloat16); thw = b['thw'].to(m.device); ones = torch.ones_like(ids)
lp = lambda x: F.log_softmax(x.float(), -1)
D = lambda x, y: (round(float((lp(x) - lp(y)).abs().max()), 4), bool((x.argmax(-1) == y.argmax(-1)).all()))
tail = lambda h: m.lm_head(h[0, a0 - 1:n - 1])
with torch.no_grad():
    ref = m(input_ids=ids, pixel_values=pv, image_grid_thw=thw, attention_mask=ones, use_cache=False).logits[0, a0 - 1:n - 1]
    # HF pieces
    hf_pos, _ = m.model.get_rope_index(ids, thw, None, attention_mask=ones)
    hf_img = m.model.get_image_features(pv, thw)
    hf_img = torch.cat(hf_img, 0) if isinstance(hf_img, (list, tuple)) else hf_img
    # ours
    ids_, emb, pos = p0.embed(m, b, None)
    our_img = p0.vis(m, b['pv'], b['thw'])
    hf_emb = m.model.get_input_embeddings()(ids).clone()
    hf_emb = hf_emb.masked_scatter((ids == p0.IMG)[..., None].expand_as(hf_emb), hf_img.to(hf_emb.dtype))
    print('POS equal', bool(torch.equal(pos, hf_pos)), tuple(pos.shape), tuple(hf_pos.shape))
    print('IMG feat', tuple(our_img.shape), tuple(hf_img.shape), 'maxdiff', float((our_img.float() - hf_img.float()).abs().max()))
    print('EMB maxdiff', float((emb.float() - hf_emb.float()).abs().max()))
    c4 = p0.causal4(ids)
    rows = {
        'A m.model(input_ids, 2D ones)': m.model(input_ids=ids, pixel_values=pv, image_grid_thw=thw, attention_mask=ones, use_cache=False).last_hidden_state,
        'B m.model(our emb, our pos, 2D ones)': m.model(inputs_embeds=emb, position_ids=pos, attention_mask=ones, use_cache=False).last_hidden_state,
        'C language_model(our emb, our pos, 2D ones)': m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=ones, use_cache=False).last_hidden_state,
        'D language_model(our emb, our pos, causal4 bool)': m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=c4, use_cache=False).last_hidden_state,
        'E language_model(our emb, our pos, causal4 float additive)': m.model.language_model(
            inputs_embeds=emb, position_ids=pos, use_cache=False,
            attention_mask=torch.zeros(c4.shape, dtype=torch.bfloat16, device=m.device).masked_fill(~c4, torch.finfo(torch.bfloat16).min)).last_hidden_state,
        'F language_model(our emb, our pos, None)': m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=None, use_cache=False).last_hidden_state,
        'G language_model(our emb, 4-row pos [text;t;h;w], 2D ones)': m.model.language_model(
            inputs_embeds=emb, position_ids=torch.cat([torch.arange(n, device=m.device)[None, None], pos], 0), attention_mask=ones, use_cache=False).last_hidden_state,
    }
    for k, h in rows.items():
        print('ROW', k, '-> vs ref', D(tail(h), ref), flush=True)
    import transformers, inspect
    print('TF', transformers.__version__)
    src = inspect.getsource(type(m.model.language_model).forward)
    for l in src.splitlines():
        if 'position_ids' in l and ('ndim' in l or 'shape[0]' in l or 'text_position_ids' in l):
            print('SRC', l.strip())
