"""Master pipeline: main sweep -> ablations -> sensitivity -> tables -> figures -> QA,
run sequentially in one process so the whole multi-hour run can be launched once and
tracked as a single background job instead of six manually-sequenced ones.

Every stage is independently resumable (each run_condition result is cached to its own
file and skipped if already present -- see run_main_experiment.py/run_ablations.py/
run_sensitivity.py), so re-running this script after an interruption picks up where it
left off rather than redoing finished work.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT.parent / ".venv" / "Scripts" / "python.exe"

STAGES = [
    "run_main_experiment.py",
    "run_ablations.py",
    "run_sensitivity.py",
    "make_tables.py",
    "make_figures.py",
    "qa_check.py",
]


def main():
    t0 = time.time()
    for stage in STAGES:
        print(f"\n{'=' * 70}\nSTAGE: {stage}   (elapsed so far: {time.time() - t0:.0f}s)\n{'=' * 70}", flush=True)
        result = subprocess.run([str(PYTHON), "-u", str(ROOT / stage)], cwd=str(ROOT.parent))
        if result.returncode != 0:
            print(f"\nSTAGE {stage} FAILED with exit code {result.returncode}. Stopping pipeline.", flush=True)
            sys.exit(result.returncode)
    print(f"\nPIPELINE COMPLETE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
