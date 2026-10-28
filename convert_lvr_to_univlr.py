"""Build a UniVLR-named view of the released UniVLR-Stage2-7B checkpoint.

The released checkpoint still uses the internal LVR naming (QwenWithLVR,
lvr_* config fields, <|lvr_*|> tokens, lvr_head.* weights), while the public
code reads univlr_* everywhere. This script creates a sibling directory with
renamed config/tokenizer/head weights and symlinks the large shards, so the
original download is left untouched.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nfs/fjj/Models/UniVLR-Stage2-7B"
DST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/nfs/fjj/Models/UniVLR-Stage2-7B-converted"
HEAD_FILE = "univlr_head.safetensors"

# steplatent_* fields that the code reads under univlr_* names
STEPLATENT_ALIASES = {
    "steplatent_align_layer": "univlr_align_layer",
    "steplatent_slots_per_block": "univlr_slots_per_block",
    "steplatent_target_resample_mode": "univlr_target_resample_mode",
    "steplatent_a2_replay_mode": "univlr_replay_mode",
}
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

os.makedirs(DST, exist_ok=True)

# 1) config: rename lvr fields, keep steplatent fields and add univlr aliases
with open(os.path.join(SRC, "config.json"), encoding="utf-8") as f:
    cfg = json.load(f)
new_cfg = {}
for key, value in cfg.items():
    if key.startswith("lvr_"):
        key = "univlr_" + key[len("lvr_"):]
    elif key.startswith("loss_lvr_"):
        key = "loss_univlr_" + key[len("loss_lvr_"):]
    new_cfg[key] = value
for old, new in STEPLATENT_ALIASES.items():
    if old in cfg and new not in new_cfg:
        new_cfg[new] = cfg[old]
new_cfg["architectures"] = ["QwenWithUniVLR"]
with open(os.path.join(DST, "config.json"), "w", encoding="utf-8") as f:
    json.dump(new_cfg, f, indent=2)

# 2) tokenizer files: <|lvr...|> -> <|univlr...|> (token ids unchanged)
for name in TOKENIZER_FILES:
    path = os.path.join(SRC, name)
    if not os.path.exists(path):
        continue
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(os.path.join(DST, name), "w", encoding="utf-8") as f:
        f.write(text.replace("<|lvr", "<|univlr"))

# 3) copy remaining small files as-is
for name in os.listdir(SRC):
    src_path = os.path.join(SRC, name)
    if (
        os.path.isfile(src_path)
        and name not in TOKENIZER_FILES
        and name not in ("config.json", "model.safetensors.index.json")
        and not name.endswith(".safetensors")
        and not name.endswith(".incomplete")
    ):
        shutil.copy2(src_path, os.path.join(DST, name))

# 4) weights: symlink original shards, re-save lvr_head.* as univlr_head.*
with open(os.path.join(SRC, "model.safetensors.index.json"), encoding="utf-8") as f:
    index = json.load(f)
weight_map = index["weight_map"]
head_keys = [k for k in weight_map if k.startswith("lvr_head.")]
head_tensors = {}
for shard in sorted({weight_map[k] for k in head_keys}):
    with safe_open(os.path.join(SRC, shard), framework="pt") as f:
        for key in head_keys:
            if weight_map[key] == shard:
                head_tensors["univlr_head." + key[len("lvr_head."):]] = f.get_tensor(key)
save_file(head_tensors, os.path.join(DST, HEAD_FILE), metadata={"format": "pt"})

for shard in sorted(set(weight_map.values())):
    link = os.path.join(DST, shard)
    if not os.path.lexists(link):
        os.symlink(os.path.join(SRC, shard), link)

new_map = {k: v for k, v in weight_map.items() if not k.startswith("lvr_head.")}
new_map.update({k: HEAD_FILE for k in head_tensors})
index["weight_map"] = new_map
with open(os.path.join(DST, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)

print("converted head keys:", sorted(head_tensors))
print("univlr config keys:", sorted(k for k in new_cfg if "univlr" in k))
print("DONE ->", DST)
