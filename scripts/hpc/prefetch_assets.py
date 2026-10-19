"""Download every pinned model and metric weight for offline COE-HPC3 jobs (coe-hpc1 only).

    source env/site.sh
    nice -n 19 python scripts/hpc/prefetch_assets.py              # download + verify
    python scripts/hpc/prefetch_assets.py --verify-only           # re-hash, no network

Pins come from eldm.assets. HF snapshots go to $HF_HOME and metric weights to
$TORCH_HOME/hub/checkpoints. Every file is checked against its hash after download, so
reruns only verify. Last, FID, KID and LPIPS are built and run on CPU with the network
blocked, which proves they load from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from fnmatch import fnmatch
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true", help="Re-hash cached assets; never download")
    return ap.parse_args()


ARGS = parse_args()
# env/site.sh makes everything offline; downloads need the network for this process only.
# Set before importing huggingface_hub, which reads these at import time.
if not ARGS.verify_only:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.umask(0o002)

import torch  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from huggingface_hub.constants import HF_XET_CACHE  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402

from eldm import assets  # noqa: E402


def git_blob_sha1(path: Path) -> str:
    """The git blob id, which is the Hub's hash for small (non-LFS) files."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def hub_hashes(asset: assets.HFAsset) -> dict[str, tuple[str, str]]:
    """Hashes the Hub publishes for each wanted file at the pinned revision:
    ("sha256", LFS sha256) for large files, ("git", blob id) for small ones."""
    wanted = {}
    for entry in HfApi().list_repo_tree(asset.repo, revision=asset.revision, recursive=True):
        if isinstance(entry, RepoFile) and any(fnmatch(entry.path, p) for p in asset.allow):
            wanted[entry.path] = ("sha256", entry.lfs.sha256) if entry.lfs else ("git", entry.blob_id)
    return wanted


def fetch_hf(name: str, asset: assets.HFAsset) -> str:
    if ARGS.verify_only:
        n, size = assets.verify_hf(name)
        return f"{n} files, {size / 1e9:.2f} GB, sha256 pins ok"

    snapshot = Path(snapshot_download(asset.repo, revision=asset.revision,
                                      allow_patterns=list(asset.allow), max_workers=4))
    # Check every file against the Hub's own hashes, then against the pins in eldm.assets.
    sha256, size = {}, 0
    for rel, (kind, expected) in sorted(hub_hashes(asset).items()):
        path = snapshot / rel
        if not path.exists():
            raise RuntimeError(f"{name}: {rel} was not downloaded")
        sha256[rel] = assets.sha256_file(path)
        actual = sha256[rel] if kind == "sha256" else git_blob_sha1(path)
        if actual != expected:
            raise RuntimeError(f"{name}: {rel} has {kind} {actual}, the Hub says {expected}")
        size += path.stat().st_size
    if dict(asset.sha256) != sha256:
        pins = "".join(f'\n            "{rel}": "{digest}",' for rel, digest in sha256.items())
        raise RuntimeError(f"{name}: sha256 pins in eldm.assets differ from the Hub-verified files:"
                           f"\n        sha256={{{pins}\n        }},")
    return f"{len(sha256)} files, {size / 1e9:.2f} GB, match the Hub and the sha256 pins"


def fetch_file(name: str, asset: assets.FileAsset) -> str:
    dst = assets.torch_hub_file(name)
    if not dst.exists() and not ARGS.verify_only:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Writes to a temporary file, checks the hash prefix, then renames into place.
        torch.hub.download_url_to_file(asset.url, str(dst), hash_prefix=asset.hash_prefix, progress=False)
    return f"{assets.verify_torch_hub(name) / 1e9:.2f} GB, sha256 pin ok"


def block_network() -> None:
    """Make any network access fail at once instead of reaching the internet or hanging."""
    import socket

    def refuse(*args, **kwargs):
        raise RuntimeError("network access attempted; the asset must come from the local cache")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]


def warm_up_metrics() -> str:
    """Build and run FID, KID and LPIPS on 2 tiny CPU images with the network blocked."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    block_network()
    g = torch.Generator().manual_seed(0)
    real = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=g)
    fake = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=g)
    results = []
    for metric in (FrechetInceptionDistance(feature=2048), KernelInceptionDistance(subset_size=2)):
        metric.update(real, real=True)
        metric.update(fake, real=False)
        value = metric.compute()
        results.append(f"{type(metric).__name__} {float(value[0] if isinstance(value, tuple) else value):.3g}")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
    results.append(f"LPIPS {float(lpips(real.float() / 255, fake.float() / 255)):.3g}")
    return ", ".join(results)


def main() -> int:
    steps = [(f"hf:{n}", lambda n=n, a=a: fetch_hf(n, a)) for n, a in assets.HF.items()]
    steps += [(f"torch_hub:{n}", lambda n=n, a=a: fetch_file(n, a)) for n, a in assets.TORCH_HUB.items()]
    steps.append(("metrics offline (FID, KID, LPIPS)", warm_up_metrics))

    failures = 0
    for label, step in steps:
        try:
            print(f"  ok    {label}: {step()}", flush=True)
        except Exception as e:  # report every asset, not just the first failure
            failures += 1
            print(f"  FAIL  {label}: {type(e).__name__}: {e}", flush=True)

    # hf_xet may keep a chunk cache next to the snapshots. Offline jobs never need it, but
    # it cannot be removed here: on NFS, this process's open xet log files block it.
    xet = Path(HF_XET_CACHE)
    if xet.exists():
        size = sum(f.stat().st_size for f in xet.rglob("*") if f.is_file())
        print(f"  info  hf_xet cache {xet}: {size / 1e9:.2f} GB (safe to delete when no download runs)")

    print(f"{failures} asset check(s) failed." if failures else "All assets present and verified.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
