"""End-to-end pipeline: raw TSVs -> both submission files.

    python run_pipeline.py --version final [--train-frac 0.5] [--blend]

Steps (each reads/writes artifacts/ so any step can be re-run on its own):
  1. convert raw TSVs to parquet                          (io_utils)
  2. learn the native-script -> Latin token map on train  (translit)
  3. blocking candidates for every split/country          (build_candidates; pool 600 -> top-20 + reverse top-3)
  4. pair features: deterministic 50% train S1 sample, all test S1   (features)
  5. extra features: legal forms, house-number relation, twins, raw-address parts (extra_features)
  6. XGBoost (CUDA) GroupKFold training + OOF evaluation (train_gpu --extra)
  7. optional: CatBoost + OOF-weighted blend             (ensemble; --blend)
  8. test inference, one-owner + expected-F0.5 selection, write/validate TSVs (predict)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    """Run one pipeline step as a subprocess and stop on failure."""
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", *args], cwd=SRC, check=True)
    print(f"=== done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    """Execute every pipeline step in order."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="final")
    ap.add_argument("--model", default="xgb_final")
    ap.add_argument("--train-frac", default="0.5")
    ap.add_argument("--lr", default="0.08")
    ap.add_argument("--blend", action="store_true", help="also train CatBoost and predict with the OOF blend")
    a = ap.parse_args()
    run(["-c", "from io_utils import convert_all_to_parquet; convert_all_to_parquet()"])
    run(["translit.py"])
    run(["build_candidates.py", "--split", "all"])
    run(["features.py", "--split", "train", "--frac", a.train_frac])
    run(["extra_features.py", "--split", "train"])
    run(["features.py", "--split", "test"])
    run(["extra_features.py", "--split", "test"])
    run(["train_gpu.py", "--name", a.model, "--lr", a.lr, "--folds", "3", "--extra"])
    if a.blend:
        run(["ensemble.py", "cat", "--name", f"{a.model}_cat"])
        run(["ensemble.py", "blend", "--names", f"{a.model},{a.model}_cat", "--out", f"{a.model}_blend"])
        run(["predict.py", "--name", f"{a.model}_blend", "--version", a.version, "--blend"])
    else:
        run(["predict.py", "--name", a.model, "--version", a.version])


if __name__ == "__main__":
    main()
