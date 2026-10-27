import os
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A stand-in job script that sources the prelude and reports what the job sees.
JOB = textwrap.dedent(f"""\
    set -euo pipefail
    CLONE="{ROOT}"
    source "${{CLONE}}/scripts/hpc/job_prelude.sh"
    for v in JUNK LD_LIBRARY_PATH PYTHONPATH CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \\
             SLURM_JOB_ID OMP_NUM_THREADS MKL_NUM_THREADS STAGE; do
        echo "@$v=${{!v:-unset}}"
    done
    [[ -d "$STAGE" ]] && echo "@STAGE_EXISTS=yes"
    echo "@ARGS=$(printf '%s|' "$@")"
""")


def run_job(tmp_path: Path, **env) -> dict[str, str]:
    script = tmp_path / "job.sh"
    script.write_text(JOB)
    base = {"HOME": os.environ.get("HOME", str(tmp_path)), "USER": "tester", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(["bash", str(script), "a", "b c"], env={**base, **env},
                         capture_output=True, text=True, check=True).stdout
    return dict(line[1:].split("=", 1) for line in out.splitlines() if line.startswith("@"))


def test_prelude_cleans_env_and_keeps_job_vars(tmp_path):
    seen = run_job(
        tmp_path,
        JUNK="leak", LD_LIBRARY_PATH="/module/lib", PYTHONPATH="/module/site-packages",
        CUDA_VISIBLE_DEVICES="2", SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="3", TMPDIR=str(tmp_path),
    )
    # Nothing from the submitting shell survives...
    assert seen["JUNK"] == "unset" and seen["LD_LIBRARY_PATH"] == "unset"
    assert seen["PYTHONPATH"] == f"{ROOT}/src"
    # ...except what the job needs: its GPU, Slurm's variables and the arguments.
    assert seen["CUDA_VISIBLE_DEVICES"] == "2" and seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert seen["SLURM_JOB_ID"] == "12345"
    assert seen["OMP_NUM_THREADS"] == seen["MKL_NUM_THREADS"] == "3"
    assert seen["ARGS"] == "a|b c|"
    # A per-job stage directory under TMPDIR, removed when the job exits.
    assert seen["STAGE"] == f"{tmp_path}/eldm-12345" and seen["STAGE_EXISTS"] == "yes"
    assert not (tmp_path / "eldm-12345").exists()


def test_prelude_runs_outside_slurm(tmp_path):
    # `bash script` on coe-hpc1: no Slurm, GPU or TMPDIR variables to keep. Bash < 4.4
    # treats the empty keep-list as unbound under set -u; this must still work.
    seen = run_job(tmp_path, JUNK="leak")
    assert seen["JUNK"] == "unset" and seen["CUDA_VISIBLE_DEVICES"] == "unset"
    assert seen["OMP_NUM_THREADS"] == "unset"
    assert seen["STAGE"].startswith("/tmp/eldm-local-") and seen["STAGE_EXISTS"] == "yes"
    assert not Path(seen["STAGE"]).exists()
    assert seen["ARGS"] == "a|b c|"
