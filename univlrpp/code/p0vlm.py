"""VLMEvalKit model class for P0 (slot-interface SFT of Qwen2.5-VL-7B-Instruct), official-wrapper style.
Prompt = Qwen2VLPromptMixin (as Qwen2VLChat / Qwen2VLChatUniVLR), default system prompt, image budget max_pixels
4,014,080 = 5120 tokens, greedy, max_new_tokens 512, no answer scaffold; VLMEvalKit's exact matching parses the output.
The assistant turn starts with the latent span <|univlr_start|> K x <|univlr|> <|univlr_end|><|univlr_latent_end|>, slots =
ViT(first image at the K-token grid) (the deployment condition). masked=True: strict slot-only mask on the prefill and on
every generated token (the answer sees the image only through the slots); masked=False: plain causal attention.
Register with run_vlm_p0.py. `python p0vlm.py --smoke <ckpt>`: 4 V* items in both modes.
"""
import os, sys, warnings
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, '/mnt/cpfs/jieduan/univlr/experiments/p0_20261111/code')
import torch  # noqa: E402
import p0  # noqa: E402
from vlmeval.vlm.base import BaseModel  # noqa: E402
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin  # noqa: E402


class P0VLM(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path, masked=True, k=384, max_new_tokens=512, min_pixels=p0.MINP, max_pixels=p0.MAXP, **kwargs):
        super().__init__(use_custom_prompt=True)
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.masked, self.k, self.max_new = bool(masked), k, max_new_tokens
        self.proc = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
        self.proc.tokenizer.add_tokens(p0.TOKS, special_tokens=True)  # no-op on a P0 checkpoint
        assert self.proc.tokenizer.convert_tokens_to_ids(p0.TOKS) == [p0.START, p0.SLOT, p0.LEND, p0.END]
        p0.IMG = self.proc.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        dev = int(os.environ.get('LOCAL_RANK', 0))  # torchrun: one replica per rank (run.py shards the dataset)
        torch.cuda.set_device(dev)
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                    device_map={'': dev}).eval()
        tk = self.proc.tokenizer
        self.eos = {tk.convert_tokens_to_ids('<|im_end|>'), tk.eos_token_id}

    def generate_inner(self, message, dataset=None):
        from PIL import Image
        imgs = [x['value'] for x in message if x['type'] == 'image']
        if not imgs:
            raise ValueError('P0VLM needs an image')
        if len(imgs) > 1:
            warnings.warn(f'P0VLM uses the first of {len(imgs)} images')
        content, seen = [], False
        for x in message:
            if x['type'] == 'image':
                if not seen:
                    content.append({'type': 'image'}); seen = True
            else:
                content.append({'type': 'text', 'text': x['value']})
        im = Image.open(imgs[0]).convert('RGB')
        text = self.proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True) + p0.span(self.k)
        x = self.proc(text=[text], images=[im], return_tensors='pt')
        zb, tb = p0.zimage(self.proc, im, self.k)
        return self.decode({'ids': x['input_ids'], 'pv': x['pixel_values'], 'thw': x['image_grid_thw'], 'zb': zb, 'tb': tb})

    @torch.no_grad()
    def decode(self, b):
        from transformers import DynamicCache
        m = self.m
        ids, emb, pos = p0.embed(m, b, p0.vis(m, b['zb'], b['tb']))
        n = ids.shape[1]; img = ids[0] == p0.IMG
        am = p0.mask4(ids) if self.masked else p0.causal4(ids)
        out = m.model.language_model(inputs_embeds=emb, position_ids=pos, attention_mask=am, past_key_values=DynamicCache(), use_cache=True)
        cache, lg, last = out.past_key_values, m.lm_head(out.last_hidden_state[0, -1]), pos[:, :, -1:]
        toks = []
        for t in range(self.max_new):
            v = int(lg.argmax())
            if v in self.eos:
                break
            toks.append(v)
            row = torch.ones(n + t + 1, dtype=torch.bool, device=m.device)
            if self.masked:
                row[:n][img] = False
            out = m.model.language_model(inputs_embeds=m.model.get_input_embeddings()(torch.tensor([[v]], device=m.device)),
                                         position_ids=last + t + 1, attention_mask=row[None, None, None], past_key_values=cache, use_cache=True)
            cache, lg = out.past_key_values, m.lm_head(out.last_hidden_state[0, -1])
        return self.proc.tokenizer.decode(toks, skip_special_tokens=True).strip()


if __name__ == '__main__' and len(sys.argv) == 3 and sys.argv[1] == '--smoke':
    from vlmeval.dataset import build_dataset
    ds = build_dataset('VStarBench')
    for masked in (True, False):
        mdl = P0VLM(sys.argv[2], masked=masked); mdl.set_dump_image(ds.dump_image)  # run.py does this itself
        for i in range(4):
            line = ds.data.iloc[i]
            msg = mdl.build_prompt(line, dataset='VStarBench') if mdl.use_custom_prompt('VStarBench') else ds.build_prompt(line)
            print('SMOKE masked', masked, 'out', repr(mdl.generate(msg, dataset='VStarBench'))[:160], 'gold', line['answer'], flush=True)
        del mdl; torch.cuda.empty_cache()
    print('SMOKE_OK', flush=True)
