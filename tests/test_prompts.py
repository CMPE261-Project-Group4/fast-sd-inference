import importlib.util
import json
from pathlib import Path

import pytest

from eldm.prompts import (
    Prompt,
    build_eval_set,
    load_prompts,
    normalize_text,
    read_partiprompts,
    sample_coco,
    save_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prepare_prompts", ROOT / "scripts/prepare_prompts.py")
prepare_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_prompts)

HEADER = "Prompt\tCategory\tChallenge\tNote\n"


def write_parti(path: Path, lines: list[str]) -> Path:
    path.write_text(HEADER + "".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_coco(path: Path, num_images: int = 20) -> Path:
    """Image i has captions "img<i> cap<j>" for j < 2 + i % 3, listed out of id order."""
    anns = [{"id": 100 * i + j, "image_id": 1000 + i, "caption": f" img{i}\ncap{j} "}
            for i in range(num_images) for j in range(2 + i % 3)]
    path.write_text(json.dumps({"annotations": anns[::-1]}))
    return path


def test_normalize_text():
    assert normalize_text("  a\n red\t\tcube  ") == "a red cube"


def test_partiprompts_keep_quotes_and_normalize_whitespace(tmp_path):
    tsv = write_parti(tmp_path / "p.tsv", [
        '"G I G G L E" painted on a wall\tArts\tWriting & Symbols\t',
        "  a  red cube \tAbstract\tBasic\tA note, with \"quotes\"",
    ])
    rows = read_partiprompts(tsv)
    # csv's default quoting would turn the first prompt into 'G I G G L E painted on a wall'.
    assert rows[0] == {"Prompt": '"G I G G L E" painted on a wall', "Category": "Arts",
                       "Challenge": "Writing & Symbols", "Note": ""}
    assert rows[1]["Prompt"] == "a red cube" and rows[1]["Note"] == 'A note, with "quotes"'


@pytest.mark.parametrize("text", [
    "Prompt\tCategory\tChallenge\n",                      # missing column
    HEADER + "a cube\tAbstract\tBasic\n",                  # missing field
    HEADER + "a cube\tAbstract\tBasic\tnote\textra\n",     # extra field
    HEADER + "  \tAbstract\tBasic\t\n",                    # empty prompt
])
def test_partiprompts_reject_malformed_files(tmp_path, text):
    (tmp_path / "p.tsv").write_text(text)
    with pytest.raises(ValueError):
        read_partiprompts(tmp_path / "p.tsv")


def test_sample_coco_is_deterministic_and_one_caption_per_image(tmp_path):
    captions = write_coco(tmp_path / "c.json")
    picked = sample_coco(captions, 8, seed=0)
    assert picked == sample_coco(captions, 8, seed=0)
    assert picked != sample_coco(captions, 8, seed=1)
    assert len({image_id for image_id, _ in picked}) == 8
    for image_id, caption in picked:
        i = image_id - 1000
        assert caption in {f"img{i} cap{j}" for j in range(2 + i % 3)}  # normalized, from its own image


def test_sample_coco_grows_by_appending(tmp_path):
    # Phase 9 may add COCO prompts for FID; the ones already frozen must not change.
    captions = write_coco(tmp_path / "c.json")
    assert sample_coco(captions, 20, seed=0)[:8] == sample_coco(captions, 8, seed=0)
    with pytest.raises(ValueError):
        sample_coco(captions, 21, seed=0)


def test_build_eval_set_ids_seeds_and_fields():
    coco = [(42, "a dog"), (7, "a cat")]
    parti = [{"Prompt": "bond", "Category": "Abstract", "Challenge": "Basic", "Note": "x"}]
    prompts = build_eval_set(coco, parti, base_seed=1000)
    assert [(p.id, p.seed) for p in prompts] == [(0, 1000), (1, 1001), (2, 1002)]
    assert prompts[0] == Prompt(0, "a dog", "coco", "coco", 1000, coco_image_id=42)
    assert prompts[2] == Prompt(2, "bond", "parti", "Abstract", 1002, challenge="Basic")


def test_old_jsonl_without_challenge_still_loads(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({"id": 0, "prompt": "a cat", "source": "coco", "category": "coco",
                                "seed": 1000, "coco_image_id": 5}) + "\n")
    assert load_prompts(path) == [Prompt(0, "a cat", "coco", "coco", 1000, 5)]


def test_save_and_load_round_trip(tmp_path):
    prompts = [Prompt(0, "a dog", "coco", "coco", 1000, 42),
               Prompt(1, '"Hi" — on a sign', "parti", "Arts", 1001, challenge="Writing & Symbols")]
    save_prompts(prompts, tmp_path / "p.jsonl")
    assert load_prompts(tmp_path / "p.jsonl") == prompts


def test_prepare_prompts_builds_the_full_set(tmp_path, capsys):
    tsv = write_parti(tmp_path / "p.tsv", ["bond\tAbstract\tBasic\t", "two cats\tAnimals\tQuantity\t"])
    out = tmp_path / "eval.jsonl"
    argv = ["--coco-captions", str(write_coco(tmp_path / "c.json")), "--parti-tsv", str(tsv),
            "--num-coco", "5", "--out", str(out), "--skip-token-count"]
    assert prepare_prompts.main(argv) == 0
    prompts = load_prompts(out)
    assert [p.source for p in prompts] == ["coco"] * 5 + ["parti"] * 2
    assert [p.id for p in prompts] == list(range(7)) and all(p.seed == 1000 + p.id for p in prompts)
    assert "Wrote 7 prompts" in capsys.readouterr().out
    first = out.read_bytes()
    prepare_prompts.main(argv)
    assert out.read_bytes() == first  # byte-for-byte reproducible
