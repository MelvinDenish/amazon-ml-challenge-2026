"""Run one pipeline job inside a Kaggle notebook (private dataset, no external data).

Private Kaggle dataset layout (uploaded once):
    data/*.parquet              <- work/artifacts/data/*.parquet
    translit_map.parquet        <- work/artifacts/translit_map.parquet
    src/*.py                    <- this folder
    feats/*.parquet  (optional) <- work/artifacts/feats/*.parquet, for GPU training jobs

Notebook cell (internet ON only for pip install of rapidfuzz/polars/xgboost):
    !pip install -q rapidfuzz polars xgboost
    !python /kaggle/input/<dataset-name>/src/kaggle_runner.py --job cands --split train --countries India

Jobs
  cands     blocking candidates      -> /kaggle/working/artifacts/cands/*.parquet
  feats     pair features            -> /kaggle/working/artifacts/feats/*.parquet
  train     XGBoost GPU training     -> /kaggle/working/artifacts/{models,oof}/
Download the produced parquet/model files from the notebook's Output tab and
copy them into the same folders under work/artifacts/ on the laptop.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent
WORK = Path("/kaggle/working/artifacts")


def stage_inputs() -> None:
    """Copy (or link) the dataset's parquet files into the artifact layout the code expects."""
    (WORK / "data").mkdir(parents=True, exist_ok=True)
    for f in (INPUT / "data").glob("*.parquet"):
        dest = WORK / "data" / f.name
        if not dest.exists():
            os.symlink(f, dest)
    if (INPUT / "translit_map.parquet").exists():
        shutil.copy(INPUT / "translit_map.parquet", WORK / "translit_map.parquet")
    if (INPUT / "feats").exists():
        (WORK / "feats").mkdir(exist_ok=True)
        for f in (INPUT / "feats").glob("*.parquet"):
            dest = WORK / "feats" / f.name
            if not dest.exists():
                os.symlink(f, dest)


def main() -> None:
    """Stage inputs, then run the requested pipeline step with artifacts in /kaggle/working."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["cands", "feats", "train"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--countries", default="")
    ap.add_argument("--frac", default="1.0")
    ap.add_argument("--name", default="xgb_kaggle")
    a = ap.parse_args()
    stage_inputs()
    env = {**os.environ, "BER_ARTIFACT_DIR": str(WORK)}
    if a.job == "cands":
        cmd = ["build_candidates.py", "--split", a.split]
    elif a.job == "feats":
        cmd = ["features.py", "--split", a.split, "--frac", a.frac]
    else:
        cmd = ["train_gpu.py", "--name", a.name, "--folds", "3"]
    if a.countries and a.job != "train":
        cmd += ["--countries", a.countries]
    subprocess.run([sys.executable, "-X", "utf8", *cmd], cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
