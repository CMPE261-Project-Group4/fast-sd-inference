# Frozen evaluation prompts

Every run uses these files: the same prompts, ids and seeds in every config, so images can be compared one to one across runs. **Don't edit them by hand.** They were frozen in Phase 4 (2026-10-28). Regenerating them must reproduce them byte for byte (see [Checking the files](#checking-the-files)); any intended change needs the team's agreement and a note in [docs/hpc/PROGRESS.md](../../docs/hpc/PROGRESS.md).

## Files

| File | Rows | Contents | Used for |
|---|---|---|---|
| `eval_prompts.jsonl` | 3,632 | ids 0–1999: COCO captions; ids 2000–3631: all of PartiPrompts | the full sweeps; `configs/base.yaml` points here |
| `subsets/debug.jsonl` | 50 | ids 0–49 (COCO) | debug runs and Phase 5's determinism check |
| `subsets/pilot.jsonl` | 200 | COCO ids 0–99 + 100 PartiPrompts, stratified by Challenge | the Phase 6 pilot (gate G2) |
| `subsets/parti_550.jsonl` | 2,550 | all 2,000 COCO + 550 PartiPrompts, stratified by Challenge | the sweeps, if gate G2 picks the smaller set |
| `manual_prompts.yaml` | 25 | hand-written smoke-test prompts; **not** part of the evaluation | `scripts/hpc/gpu_check.py` |

The subsets are rows of `eval_prompts.jsonl`, unchanged, so a prompt has the same id and seed everywhere, and `debug` ⊂ `pilot` ⊂ `parti_550`. To run on a subset, set `prompts_file: data/prompts/subsets/<name>.jsonl` in the experiment file.

## Fields

| Field | Meaning |
|---|---|
| `id` | 0–3631, the image's file name: `images/{id:04d}.png` |
| `prompt` | the text, with whitespace collapsed to single spaces |
| `source` | `coco` or `parti` |
| `category` | PartiPrompts' Category (12 of them); `coco` for COCO |
| `challenge` | PartiPrompts' Challenge (11 of them); null for COCO |
| `seed` | the generation seed, `1000 + id` |
| `coco_image_id` | the COCO val2017 image the caption describes: the FID/KID reference `val2017/{coco_image_id:012d}.jpg`. Null for PartiPrompts |

## How they're built

On coe-hpc1, from the repo root (about 3 s):
```bash
source env/site.sh
python scripts/prepare_prompts.py
```

The defaults are the frozen settings: `--num-coco 2000 --sample-seed 0 --base-seed 1000`, reading `$ELDM_DATASETS/coco/annotations/captions_val2017.json` and `$ELDM_DATASETS/partiprompts/PartiPrompts.tsv` (their checksums are in [data/CHECKSUMS.txt](../CHECKSUMS.txt)).

- **COCO.** All 5,000 val2017 images are put in one seeded order (`random.Random(0)` shuffles the sorted image ids), and then one of each image's 5–7 captions is picked, in that order. The set is the first 2,000. Adding more later (gate G4: 3,000 more for FID, appended as ids 3632+) takes the next images in the same order, and leaves these 2,000 and their captions unchanged.
- **PartiPrompts.** All 1,632, in file order. The TSV is parsed literally, without csv quoting. Python's default quoting would strip the quotes from `"G I G G L E" painted in thick colorful lettering…` (id 3560).
- **Whitespace.** Runs of whitespace are collapsed to one space and the ends are stripped. This changed 3,814 of the 26,646 source texts (mostly trailing spaces), and none of their CLIP token ids.
- **Subsets.** The PartiPrompts part is split across Challenges in proportion to their sizes, with largest-remainder rounding (ties by name). Each Challenge gets its own seeded order, from the sample seed and its name, and the first k are taken. That's why the smaller subsets nest inside the larger.

## Counts

| Challenge | Full | pilot | parti_550 |
|---|---|---|---|
| Fine-grained Detail | 312 | 19 | 105 |
| Basic | 272 | 17 | 92 |
| Simple Detail | 232 | 14 | 78 |
| Style & Format | 206 | 13 | 69 |
| Imagination | 150 | 9 | 50 |
| Complex | 113 | 7 | 38 |
| Writing & Symbols | 91 | 6 | 31 |
| Quantity | 90 | 5 | 30 |
| Perspective | 70 | 4 | 24 |
| Linguistic Structures | 61 | 4 | 21 |
| Properties & Positioning | 35 | 2 | 12 |
| **PartiPrompts** | **1,632** | **100** | **550** |
| COCO | 2,000 | 100 | 2,000 |

**Over CLIP's 77 tokens:** one prompt, id 2358 (84 tokens, "Oil-on-canvas painting of a blue night sky with roiling energy…"). It's kept: the pipeline truncates it to 77 tokens the same way in every config. The longest COCO caption has 58.

## Checking the files

`tests/test_prompts.py` checks the counts, ids, seeds and subsets. It also regenerates the files and compares them byte for byte with these, wherever the shared datasets exist. By hand:
```bash
cd data/prompts && sha256sum eval_prompts.jsonl subsets/*.jsonl
```
must print:
```
c51d406b672554d1289c6adbf580fb92f8a5aa379cc660e4426fd8e70dec805d  eval_prompts.jsonl
77a7c73c8e841b164963c991f2f0cae130c7da64a0d3cfaa68a8c9e37eede6cb  subsets/debug.jsonl
3b2949d534a8f6cb4b879a5d41f938a38cd0e01c1a5c1f3fd1361bb4d0eb79a0  subsets/parti_550.jsonl
1df157551a58c05d9d2ee2ec82d60335c12c07bf83aeaf8adfcd6711b4483d4f  subsets/pilot.jsonl
```

## Sources and licenses

- **COCO 2017 val captions** (`annotations_trainval2017.zip`, images.cocodataset.org). The annotations are © the COCO Consortium, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Only captions and image ids are copied here; the images stay on shared storage.
- **PartiPrompts** ([google-research/parti](https://github.com/google-research/parti), `PartiPrompts.tsv` at commit `5a657978134374ce28973948331b319adef164bd`), under the Apache License 2.0.
