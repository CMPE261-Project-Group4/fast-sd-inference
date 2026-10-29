import csv
import importlib.util
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from eldm import assets
from eldm.metrics import ClipScorer, lpips_to_baseline
from eldm.prompts import Prompt, save_prompts

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evaluate_quality", ROOT / "scripts/evaluate_quality.py")
evaluate_quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_quality)

RED, GREEN, YELLOW = (255, 0, 0), (0, 255, 0), (255, 255, 0)
TEXT_EMB = {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "not red": [-1.0, 0.0, 0.0]}


def fake_scorer(batch_size: int = 32, calls: list | None = None) -> ClipScorer:
    """Images embed as their mean RGB colour, prompts as TEXT_EMB's vectors."""
    def encode_images(images):
        if calls is not None:
            calls.append(len(images))
        return torch.tensor(np.array([np.asarray(im, np.float32).mean(axis=(0, 1)) / 255 for im in images]))

    return ClipScorer(encode_images, lambda prompts: torch.tensor([TEXT_EMB[p] for p in prompts]), batch_size)


def solid(path: Path, colour) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)
    return path


def test_clip_score_formula_clamp_and_order(tmp_path):
    images = [solid(tmp_path / f"{i}.png", c) for i, c in enumerate([RED, GREEN, YELLOW, RED])]
    calls = []
    scores = fake_scorer(batch_size=3, calls=calls).scores(images, ["red", "red", "red", "not red"])
    # 100 * max(cos, 0): same direction 100, orthogonal 0, 45 degrees 70.7, opposite clamped to 0.
    assert scores == pytest.approx([100.0, 0.0, 100 / 2**0.5, 0.0])
    assert calls == [3, 1]  # batched, and the order survives the batch boundary


def test_clip_score_needs_one_prompt_per_image(tmp_path):
    with pytest.raises(ValueError, match="2 images but 1 prompts"):
        fake_scorer().scores([solid(tmp_path / "a.png", RED)] * 2, ["red"])
    with pytest.raises(ValueError, match="unknown CLIP model"):
        ClipScorer.load("vitb", "cpu")


def make_run(root: Path, name: str, colours: list) -> Path:
    prompts = [Prompt(i, text, "manual", "objects", 1000 + i) for i, text in enumerate(["red", "green", "red"])]
    save_prompts(prompts, root / "prompts.jsonl")
    run = root / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"name": name, "prompts_file": str(root / "prompts.jsonl"),
                                                     "num_prompts": None}))
    for p, colour in zip(prompts, colours):
        solid(run / "images" / f"{p.id:04d}.png", colour)
    return run


def args(**kw) -> Namespace:
    return Namespace(**{"baseline": None, "fid_ref": None, **kw})


def test_evaluate_writes_means_and_per_image_rows(tmp_path):
    run = make_run(tmp_path, "fp16", [RED, YELLOW, GREEN])
    scorers = {"vitg": fake_scorer(), "vitl": fake_scorer()}
    result, rows = evaluate_quality.evaluate(run, args(), "cpu", scorers)
    clip = [100.0, 100 / 2**0.5, 0.0]
    assert result == pytest.approx({"name": "fp16", "num_images": 3, "clip_vitg": sum(clip) / 3,
                                    "clip_vitl": sum(clip) / 3})
    evaluate_quality.write_results(run, result, rows)
    assert json.loads((run / "quality.json").read_text()) == result
    with open(run / "quality_per_image.csv", newline="") as f:
        table = list(csv.reader(f))
    assert table[0] == ["id", "clip_vitg", "clip_vitl"]
    assert [int(r[0]) for r in table[1:]] == [0, 1, 2]
    assert [float(r[1]) for r in table[1:]] == pytest.approx(clip)


def test_evaluate_lpips_against_the_baseline(tmp_path, monkeypatch):
    base = make_run(tmp_path, "baseline", [RED, GREEN, RED])
    run = make_run(tmp_path, "fp16", [RED, YELLOW, GREEN])
    seen = {}

    def fake_lpips(images, baseline, device):
        seen.update(images=[p.parent.parent.name for p in images], baseline=[p.name for p in baseline])
        return [0.0, 0.2, 0.4]

    monkeypatch.setattr(evaluate_quality, "lpips_to_baseline", fake_lpips)
    result, rows = evaluate_quality.evaluate(run, args(baseline=str(base)), "cpu", {"vitg": fake_scorer()})
    assert seen == {"images": ["fp16"] * 3, "baseline": ["0000.png", "0001.png", "0002.png"]}
    assert result["lpips_to_baseline"] == pytest.approx(0.2)
    assert [r["lpips_to_baseline"] for r in rows] == [0.0, 0.2, 0.4]
    assert list(rows[0]) == ["id", "clip_vitg", "lpips_to_baseline"]


def test_evaluate_refuses_missing_images(tmp_path):
    base = make_run(tmp_path, "baseline", [RED, GREEN, RED])
    run = make_run(tmp_path / "b", "fp16", [RED, YELLOW, GREEN])
    (run / "images" / "0001.png").unlink()
    with pytest.raises(FileNotFoundError, match="1 images missing"):
        evaluate_quality.evaluate(run, args(), "cpu", {"vitg": fake_scorer()})
    (base / "images" / "0002.png").unlink()
    with pytest.raises(FileNotFoundError, match="baseline: 1 images missing"):
        evaluate_quality.evaluate(make_run(tmp_path / "c", "ok", [RED] * 3), args(baseline=str(base)), "cpu", {})


def test_main_loads_each_model_once(tmp_path, monkeypatch):
    runs = [make_run(tmp_path / d, "baseline", [RED, GREEN, RED]) for d in ("a", "b")]
    (tmp_path / "a" / "not_a_run").mkdir()
    loaded = []
    monkeypatch.setattr(evaluate_quality.ClipScorer, "load",
                        classmethod(lambda cls, name, device: loaded.append(name) or fake_scorer()))
    evaluate_quality.main(["--runs", str(tmp_path / "*" / "*"), "--device", "cpu"])
    assert loaded == ["vitg", "vitl"]
    for run in runs:
        assert json.loads((run / "quality.json").read_text())["clip_vitg"] == pytest.approx(100.0)


# ---- slow: real models from the shared cache (pytest -m slow, by hand on coe-hpc1) ----------

DATASETS = Path(os.environ.get("ELDM_DATASETS", "/data/cmpe261-fa26/group4/datasets"))


@pytest.fixture
def coco_pairs():
    """The first two COCO val images (by id) with their first captions."""
    captions_file = DATASETS / "coco/annotations/captions_val2017.json"
    if not captions_file.exists():
        pytest.skip("needs the shared COCO dataset")
    captions = json.loads(captions_file.read_text())
    images = sorted(captions["images"], key=lambda im: im["id"])[:2]
    first = {}
    for a in sorted(captions["annotations"], key=lambda a: a["id"]):
        first.setdefault(a["image_id"], a["caption"])
    return [DATASETS / "coco/val2017" / im["file_name"] for im in images], [first[im["id"]] for im in images]


@pytest.mark.slow
def test_clip_vitl_by_hand_matches_torchmetrics(coco_pairs):
    """Our ViT-L/14 score is torchmetrics' CLIPScore, with Phase 2's adapter for transformers 5."""
    from torchmetrics.multimodal.clip_score import CLIPScore
    from transformers import CLIPModel, CLIPProcessor

    class CLIPTensorFeatures(CLIPModel):
        def get_image_features(self, *args, **kwargs):
            return super().get_image_features(*args, **kwargs).pooler_output

        def get_text_features(self, *args, **kwargs):
            return super().get_text_features(*args, **kwargs).pooler_output

    paths, captions = coco_pairs
    ours = ClipScorer.load("vitl", "cpu").scores(paths, captions)
    snapshot = assets.hf_snapshot_dir("clip_vitl14")
    model, processor = CLIPTensorFeatures.from_pretrained(snapshot), CLIPProcessor.from_pretrained(snapshot)
    metric = CLIPScore(model_name_or_path=lambda: (model, processor))
    theirs = []
    for path, caption in zip(paths, captions):
        pixels = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
        theirs.append(float(metric(pixels, caption)))
        metric.reset()
    print(f"CLIP-L by hand {ours}, torchmetrics {theirs}")
    assert ours == pytest.approx(theirs, abs=1e-3)
    assert all(15 < s < 45 for s in ours)  # a real caption of a real photo


@pytest.mark.slow
def test_lpips_per_image_mean_matches_torchmetrics(coco_pairs, tmp_path):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from eldm.metrics import load_image

    paths, _ = coco_pairs
    crops = [Image.open(p).convert("RGB").crop((0, 0, 256, 256)) for p in paths]
    a = [tmp_path / "a0.png", tmp_path / "a1.png"]
    b = [tmp_path / "b0.png", tmp_path / "b1.png"]
    for path, im in zip(a + b, crops + crops[::-1]):
        im.save(path)
    per_image = lpips_to_baseline(a, b, "cpu")
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    metric.update(torch.stack([load_image(p) for p in a]).float() / 255,
                  torch.stack([load_image(p) for p in b]).float() / 255)
    assert len(per_image) == 2 and per_image[0] == pytest.approx(per_image[1], rel=1e-4)  # LPIPS is symmetric
    assert sum(per_image) / 2 == pytest.approx(float(metric.compute()), rel=1e-5)
    assert lpips_to_baseline(a, a, "cpu") == [0.0, 0.0]
