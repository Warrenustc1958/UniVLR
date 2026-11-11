# UniVLR++ direction B: slot interface (P0/P0b) -> latent pointer + high-resolution re-look

Research code from 2026-11-10..12 (not a polished release). Paths are hard-coded to the BAAI cluster layout
(`/mnt/cpfs/jieduan/univlr/...`); adjust `R` / `MON` / `QWEN` at the top of each file to run elsewhere.

- `docs/LatentPointer_Results.md` - all results (P0/P0b SFT, gates G0'-G4', Stage B pointer, text-bbox baseline).
- `docs/LatentPointer_Design.md` - design of the latent pointer (layout, cut point, training stages, unit tests).

| file | what |
|---|---|
| `code/p0.py`, `code/p0vlm.py`, `code/run_vlm_p0.py` | P0 / P0b slot-interface SFT of Qwen2.5-VL-7B-Instruct and its VLMEvalKit model class |
| `code/score_official.py`, `code/vacc.py` | official exact-matching scoring of VLMEvalKit prediction files |
| `code/g123.py`, `code/g4p.py`, `code/g4hr.py`, `code/gates2.py` | training-free gates: GT-crop upper bound, crop drift, slot-attention localisation, pointer + re-look, layouts, unit tests U1-U4 |
| `code/g0p.py`, `code/g0probe.py` | latent interventions on the official UniVLR Stage1 / Stage2 wrapper |
| `code/tbox2.py`, `code/pair_ptrB.py` | text-bbox baseline and paired comparisons |
| `code/ptrB.py` | Stage B pointer warm-start (data build, unit tests U5-U8, training, pointer eval, end-to-end eval) |
| `scripts/run_*.sh` | cluster launchers (PPU hold/release, smoke tests, sharded jobs) |
