import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from eldm.prompts import (
    Prompt,
    allocate,
    build_eval_set,
    build_subsets,
    load_prompts,
    normalize_text,
    read_partiprompts,
    sample_coco,
    save_prompts,
    stratified_sample,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prepare_prompts", ROOT / "scripts/prepare_prompts.py")
prepare_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_prompts)

HEADER = "Prompt\tCategory\tChallenge\tNote\n"

# PartiPrompts' Challenge sizes, and the allocations the frozen subsets use (Phase 4 plan).
CHALLENGES = {"Fine-grained Detail": 312, "Basic": 272, "Simple Detail": 232, "Style & Format": 206,
              "Imagination": 150, "Complex": 113, "Writing & Symbols": 91, "Quantity": 90,
              "Perspective": 70, "Linguistic Structures": 61, "Properties & Positioning": 35}
PILOT_PARTI = dict(zip(CHALLENGES, [19, 17, 14, 13, 9, 7, 6, 5, 4, 4, 2]))
FALLBACK_PARTI = dict(zip(CHALLENGES, [105, 92, 78, 69, 50, 38, 31, 30, 24, 21, 12]))


def write_parti(path: Path, lines: list[str]) -> Path:
    path.write_text(HEADER + "".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_parti_sized(path: Path, sizes: dict[str, int]) -> Path:
    """A PartiPrompts file with sizes[c] prompts in Challenge c, the Challenges interleaved."""
    names = [c for c, size in sizes.items() for _ in range(size)]
    order = sorted(range(len(names)), key=lambda i: (i * 7919) % len(names))  # a fixed shuffle
    return write_parti(path, [f"prompt {i}\tArts\t{names[i]}\t" for i in order])


def parti_prompts(sizes: dict[str, int]) -> list[Prompt]:
    names = [c for c, size in sizes.items() for _ in range(size)]
    return [Prompt(i, f"p{i}", "parti", "Arts", 1000 + i, challenge=c) for i, c in enumerate(names)]


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


def test_allocate_matches_the_frozen_subsets():
    assert allocate(CHALLENGES, 100) == PILOT_PARTI
    assert allocate(CHALLENGES, 550) == FALLBACK_PARTI
    total = sum(CHALLENGES.values())
    for n, counts in [(100, PILOT_PARTI), (550, FALLBACK_PARTI)]:
        assert all(abs(counts[c] - n * size / total) < 1 for c, size in CHALLENGES.items())


def test_allocate_edges_and_ties():
    assert allocate({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}  # equal remainders: by name
    assert allocate({"a": 3, "b": 2}, 0) == {"a": 0, "b": 0}
    assert allocate({"a": 3, "b": 2}, 5) == {"a": 3, "b": 2}
    with pytest.raises(ValueError):
        allocate({"a": 3, "b": 2}, 6)


def test_stratified_sample_is_proportional_seeded_and_nested():
    prompts = parti_prompts(CHALLENGES)
    pilot = stratified_sample(prompts, 100, seed=0)
    fallback = stratified_sample(prompts, 550, seed=0)
    assert {c: sum(p.challenge == c for p in pilot) for c in CHALLENGES} == PILOT_PARTI
    assert {c: sum(p.challenge == c for p in fallback) for c in CHALLENGES} == FALLBACK_PARTI
    assert [p.id for p in pilot] == sorted(p.id for p in pilot)
    assert stratified_sample(prompts, 100, seed=0) == pilot
    assert stratified_sample(prompts, 100, seed=1) != pilot
    assert {p.id for p in pilot} <= {p.id for p in fallback}
    assert stratified_sample(prompts[::-1], 100, seed=0) == pilot  # input order doesn't matter


def test_stratified_sample_needs_challenges():
    with pytest.raises(ValueError):
        stratified_sample([Prompt(0, "a dog", "coco", "coco", 1000, 42)], 1, seed=0)


def test_build_subsets_reuse_full_set_rows_and_nest():
    coco = [Prompt(i, f"c{i}", "coco", "coco", 1000 + i, coco_image_id=i) for i in range(120)]
    parti = [Prompt(120 + p.id, p.prompt, "parti", p.category, 1120 + p.id, challenge=p.challenge)
             for p in parti_prompts(CHALLENGES)]
    full = coco + parti
    subsets = build_subsets(full, seed=0)
    assert {name: len(s) for name, s in subsets.items()} == {"debug": 50, "pilot": 200, "parti_550": 670}
    by_id = {p.id: p for p in full}
    for subset in subsets.values():
        assert all(by_id[p.id] == p for p in subset)  # same rows, so same ids and seeds
        assert [p.id for p in subset] == sorted(p.id for p in subset)
    ids = {name: {p.id for p in s} for name, s in subsets.items()}
    assert ids["debug"] == set(range(50))
    assert ids["debug"] <= ids["pilot"] <= ids["parti_550"]
    assert sum(p.source == "coco" for p in subsets["pilot"]) == 100
    assert sum(p.source == "coco" for p in subsets["parti_550"]) == 120


def test_prepare_prompts_writes_the_set_and_subsets(tmp_path, capsys):
    tsv = write_parti_sized(tmp_path / "p.tsv", CHALLENGES)
    out, subsets_dir = tmp_path / "eval.jsonl", tmp_path / "subsets"
    argv = ["--coco-captions", str(write_coco(tmp_path / "c.json", num_images=150)), "--parti-tsv", str(tsv),
            "--num-coco", "120", "--out", str(out), "--subsets-dir", str(subsets_dir), "--skip-token-count"]
    assert prepare_prompts.main(argv) == 0
    prompts = load_prompts(out)
    assert [p.source for p in prompts] == ["coco"] * 120 + ["parti"] * 1632
    assert [p.id for p in prompts] == list(range(1752)) and all(p.seed == 1000 + p.id for p in prompts)
    by_id = {p.id: p for p in prompts}
    sizes = {}
    for name in ("debug", "pilot", "parti_550"):
        subset = load_prompts(subsets_dir / f"{name}.jsonl")
        assert all(by_id[p.id] == p for p in subset)
        sizes[name] = len(subset)
    assert sizes == {"debug": 50, "pilot": 200, "parti_550": 670}
    assert "Wrote 1752 prompts" in capsys.readouterr().out
    first = [f.read_bytes() for f in [out, *sorted(subsets_dir.iterdir())]]
    prepare_prompts.main(argv)
    assert [f.read_bytes() for f in [out, *sorted(subsets_dir.iterdir())]] == first  # byte-for-byte


# The frozen files in data/prompts (Phase 4). These guard them against drift.
PROMPTS_DIR = ROOT / "data/prompts"
FROZEN = ["eval_prompts.jsonl", "subsets/debug.jsonl", "subsets/parti_550.jsonl", "subsets/pilot.jsonl"]
SUBSET_SIZES = {"debug": 50, "pilot": 200, "parti_550": 2550}


def test_frozen_eval_set_counts():
    full = load_prompts(PROMPTS_DIR / "eval_prompts.jsonl")
    assert [p.id for p in full] == list(range(3632))
    assert all(p.seed == 1000 + p.id for p in full)
    assert Counter(p.source for p in full) == {"coco": 2000, "parti": 1632}
    assert [p.source for p in full[:2000]] == ["coco"] * 2000
    assert len({p.coco_image_id for p in full if p.source == "coco"}) == 2000
    assert all(p.challenge is None for p in full[:2000])
    # The data doc's numbers, as PLAN.md quotes them: 90 Quantity, 91 Writing & Symbols.
    assert Counter(p.challenge for p in full[2000:]) == CHALLENGES


def test_frozen_subsets():
    full = {p.id: p for p in load_prompts(PROMPTS_DIR / "eval_prompts.jsonl")}
    subsets = {name: load_prompts(PROMPTS_DIR / f"subsets/{name}.jsonl") for name in SUBSET_SIZES}
    assert {name: len(s) for name, s in subsets.items()} == SUBSET_SIZES
    for subset in subsets.values():
        assert all(full[p.id] == p for p in subset)  # the full set's rows: same ids and seeds
    ids = {name: {p.id for p in s} for name, s in subsets.items()}
    assert ids["debug"] == set(range(50)) and ids["debug"] <= ids["pilot"] <= ids["parti_550"]
    assert Counter(p.challenge for p in subsets["pilot"] if p.challenge) == PILOT_PARTI
    assert Counter(p.challenge for p in subsets["parti_550"] if p.challenge) == FALLBACK_PARTI
    assert Counter(p.source for p in subsets["parti_550"]) == {"coco": 2000, "parti": 550}


def test_frozen_files_match_readme_checksums():
    listed = re.findall(r"^([0-9a-f]{64})  (\S+)$", (PROMPTS_DIR / "README.md").read_text(), re.MULTILINE)
    assert sorted(name for _, name in listed) == FROZEN
    for digest, name in listed:
        assert hashlib.sha256((PROMPTS_DIR / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(
    not (prepare_prompts.DATASETS / "coco/annotations/captions_val2017.json").exists()
    or not (prepare_prompts.DATASETS / "partiprompts/PartiPrompts.tsv").exists(),
    reason="needs the shared datasets ($ELDM_DATASETS)",
)
def test_frozen_files_regenerate_byte_for_byte(tmp_path):
    argv = ["--out", str(tmp_path / "eval_prompts.jsonl"), "--subsets-dir", str(tmp_path / "subsets"),
            "--skip-token-count"]
    assert prepare_prompts.main(argv) == 0
    for name in FROZEN:
        assert (tmp_path / name).read_bytes() == (PROMPTS_DIR / name).read_bytes(), name
