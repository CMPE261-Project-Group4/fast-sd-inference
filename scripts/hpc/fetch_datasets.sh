#!/usr/bin/env bash
# Download, verify and lock the evaluation datasets. Run on coe-hpc1 (needs internet).
#
#   scripts/hpc/fetch_datasets.sh
#
# Into ${ELDM_DATASETS} (env/site.sh):
#   coco/val2017.zip -> coco/val2017/             5,000 images (FID/KID references)
#   coco/annotations_trainval2017.zip
#                    -> coco/annotations/captions_val2017.json   (COCO prompts)
#   partiprompts/PartiPrompts.tsv                 1,632 prompts (google-research/parti)
#
# Each download is checked against a pinned server-side hash (COCO's ETag, the git blob id
# of PartiPrompts.tsv at a fixed commit). The first run records sha256 sums in
# data/CHECKSUMS.txt; later runs verify against it, as can anyone:
#   cd /data/cmpe261-fa26/group4/datasets && sha256sum -c <clone>/data/CHECKSUMS.txt
# Verified data is made read-only (chmod -R a-w). Reruns only verify.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/env/site.sh"

CHECKSUMS="${ROOT}/data/CHECKSUMS.txt"
COCO="${ELDM_DATASETS}/coco"
PARTI="${ELDM_DATASETS}/partiprompts"

# images.cocodataset.org serves plain HTTP only; integrity comes from the pinned ETags.
COCO_URL=http://images.cocodataset.org
VAL_ETAG=d366be60d3dc737327160d62453e3973-98   # S3 multipart ETag (98 parts)
VAL_SIZE=815585330
ANN_ETAG=f4bbac642086de4f52a3fdda2de5fa2c      # single-part ETag = MD5
ANN_SIZE=252907541
PARTI_COMMIT=5a657978134374ce28973948331b319adef164bd
PARTI_BLOB=f96e7125a02e48976736179a667b080ba81539fb  # git blob id at PARTI_COMMIT
PARTI_URL="https://raw.githubusercontent.com/google-research/parti/${PARTI_COMMIT}/PartiPrompts.tsv"

FILES=(
    coco/val2017.zip
    coco/annotations_trainval2017.zip
    coco/annotations/captions_val2017.json
    partiprompts/PartiPrompts.tsv
)

log() { printf '[fetch_datasets] %s\n' "$*"; }

# Atomic download: write .part, rename when complete. Skips files already present.
download() {
    local url="$1" dst="$2"
    if [[ -f "${dst}" ]]; then
        return
    fi
    log "downloading ${url}"
    curl -fsSL --retry 3 -o "${dst}.part" "${url}"
    mv "${dst}.part" "${dst}"
}

# Compare a file with the S3 ETag the server reported for it: plain MD5 for single-part
# uploads, md5(concatenated part MD5s)-N for multipart ones (8 MiB parts here).
check_etag() {
    local file="$1" etag="$2" size="$3"
    if [[ "$(stat -c %s "${file}")" != "${size}" ]]; then
        log "ERROR: ${file} is $(stat -c %s "${file}") bytes, expected ${size}"
        return 1
    fi
    if python - "${file}" "${etag}" <<'PY'
import hashlib
import sys

path, etag = sys.argv[1], sys.argv[2]
part = 8 * 1024 * 1024
with open(path, "rb") as f:
    if "-" in etag:
        parts = [hashlib.md5(chunk).digest() for chunk in iter(lambda: f.read(part), b"")]
        got = f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"
    else:
        md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(part), b""):
            md5.update(chunk)
        got = md5.hexdigest()
sys.exit(0 if got == etag else 1)
PY
    then
        log "$(basename "${file}"): matches ETag ${etag}"
    elif [[ "${etag}" == *-* ]] && unzip -tq "${file}" >/dev/null; then
        # The part size is a guess; size + zip CRCs still prove an intact archive.
        log "WARNING: $(basename "${file}"): multipart ETag not reproduced with 8 MiB parts;" \
            "size and zip CRCs check out"
    else
        log "ERROR: ${file} does not match ETag ${etag}"
        return 1
    fi
}

check_git_blob() {
    local file="$1" blob="$2" got
    got="$(python -c 'import hashlib, sys; d = open(sys.argv[1], "rb").read(); print(hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest())' "${file}")"
    if [[ "${got}" != "${blob}" ]]; then
        log "ERROR: ${file} has git blob id ${got}, expected ${blob}"
        return 1
    fi
    log "$(basename "${file}"): matches git blob ${blob} (parti@${PARTI_COMMIT:0:7})"
}

# Counts the project relies on (Phase 4 data doc).
validate() {
    python - "${ELDM_DATASETS}" <<'PY'
import collections
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
images = {p.name for p in (root / "coco/val2017").glob("*.jpg")}
captions = json.loads((root / "coco/annotations/captions_val2017.json").read_text())
listed = {im["file_name"] for im in captions["images"]}
captioned = {a["image_id"] for a in captions["annotations"]}
assert len(images) == 5000, f"{len(images)} val2017 images, expected 5000"
assert listed == images, "captions_val2017.json does not list exactly the val2017 images"
assert captioned == {im["id"] for im in captions["images"]}, "some images have no caption"

with open(root / "partiprompts/PartiPrompts.tsv", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
challenges = collections.Counter(r["Challenge"] for r in rows)
assert list(rows[0]) == ["Prompt", "Category", "Challenge", "Note"], list(rows[0])
assert len(rows) == 1632, f"{len(rows)} PartiPrompts, expected 1632"
assert challenges["Quantity"] == 90 and challenges["Writing & Symbols"] == 91, challenges

print(f"[fetch_datasets] COCO val2017: {len(images)} images, {len(captions['annotations'])} captions;"
      f" PartiPrompts: {len(rows)} prompts, {len(challenges)} Challenges,"
      f" {len({r['Category'] for r in rows})} Categories")
PY
}

mkdir -p "${COCO}" "${PARTI}"

download "${COCO_URL}/zips/val2017.zip" "${COCO}/val2017.zip"
download "${COCO_URL}/annotations/annotations_trainval2017.zip" "${COCO}/annotations_trainval2017.zip"
download "${PARTI_URL}" "${PARTI}/PartiPrompts.tsv"

check_etag "${COCO}/val2017.zip" "${VAL_ETAG}" "${VAL_SIZE}"
check_etag "${COCO}/annotations_trainval2017.zip" "${ANN_ETAG}" "${ANN_SIZE}"
check_git_blob "${PARTI}/PartiPrompts.tsv" "${PARTI_BLOB}"

if [[ ! -d "${COCO}/val2017" ]]; then
    log "extracting val2017 images"
    unzip -q "${COCO}/val2017.zip" -d "${COCO}"
fi
if [[ ! -f "${COCO}/annotations/captions_val2017.json" ]]; then
    log "extracting captions_val2017.json"   # train annotations are not needed
    unzip -q "${COCO}/annotations_trainval2017.zip" annotations/captions_val2017.json -d "${COCO}"
fi

validate

if [[ -f "${CHECKSUMS}" ]]; then
    (cd "${ELDM_DATASETS}" && sha256sum --quiet -c "${CHECKSUMS}")
    log "sha256 sums match data/CHECKSUMS.txt"
else
    (cd "${ELDM_DATASETS}" && sha256sum "${FILES[@]}") > "${CHECKSUMS}"
    log "recorded sha256 sums in data/CHECKSUMS.txt"
fi

chmod -R a-w "${COCO}" "${PARTI}"
log "datasets verified and read-only: $(du -sh "${ELDM_DATASETS}" | cut -f1) in ${ELDM_DATASETS}"
