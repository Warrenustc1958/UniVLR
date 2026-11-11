"""G0' probe: does the official wrapper actually emit latent tokens? Raw generations (clean_univlr_output=False) of
UniVLR-Stage1 (univlr, 1 step = s1_official) and UniVLR-Stage2 (steps, 12 = univlr_s2_1105) on 6 HRBench4K + 6 V* items;
counts <|univlr_start|>/<|univlr|>/<|univlr_end|>/<|univlr_latent_end|> and latent feed calls."""
import os, sys
R = '/mnt/cpfs/jieduan/univlr'; K = R + '/UniVLRplus/VLMEvalKit'
sys.path[:0] = [K, R + '/UniVLRplus']; os.environ.setdefault('UNIVLR_ROOT', R + '/UniVLRplus')
from vlmeval.dataset import build_dataset
from vlmeval.vlm.qwen2_vl.univlr import Qwen2VLChatUniVLR
import src.train.monkey_patch_forward_univlr_inference as mp
N = {'n': 0}; orig = mp._univlr_feed_vector
def feed(model, h):
    N['n'] += 1; return orig(model, h)
mp._univlr_feed_vector = feed
for name, path, strat, steps in (('s1', R + '/Models/UniVLR-Stage1-7B-converted', 'univlr', 1),
                                 ('s2', R + '/Models/UniVLR-Stage2-7B-converted', 'steps', 12)):
    m = Qwen2VLChatUniVLR(model_path=path, decoding_strategy=strat, univlr_steps=steps, clean_univlr_output=False,
                          max_new_tokens=512, max_pixels=4014080)
    tk = m.processor.tokenizer
    toks = ['<|univlr_start|>', '<|univlr|>', '<|univlr_end|>', '<|univlr_latent_end|>']
    for dsn in ('HRBench4K', 'VStarBench'):
        ds = build_dataset(dsn); m.set_dump_image(ds.dump_image)
        for _, line in list(ds.data.iterrows())[:6]:
            msg = m.build_prompt(line, dataset=dsn) if m.use_custom_prompt(dsn) else ds.build_prompt(line)
            N['n'] = 0; out = m.generate(message=msg, dataset=dsn)
            cnt = {t: out.count(t) for t in toks}
            print('PROBE', name, dsn, int(line['index']), 'feeds', N['n'], cnt, 'out', repr(out[-80:]), flush=True)
    del m
    import torch; torch.cuda.empty_cache()
print('PROBE_DONE', flush=True)
