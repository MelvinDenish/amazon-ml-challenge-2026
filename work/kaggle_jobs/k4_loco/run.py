"""Kaggle GPU job (Bedrock ER): everything RAM-heavy after blocking v3.

Inputs:  datasets melvindenishl/amazon-ml (data) + melvindenishl/bedrock-src (code, map),
         kernel outputs bedrock-blk3-train / bedrock-blk3-test (pool-600 candidates).
Steps:   train features (50% of S1) + extras -> XGBoost stage 1 (P100, 3-fold OOF)
         -> LOCO self-training experiment -> test features + extras -> test
         prediction + submission TSVs -> diagnose report.
Outputs: submissions/<VERSION>/*.tsv, models/<NAME>_fold*.json, oof/<NAME>.parquet,
         logs printed to the kernel log. Feature shards are deleted at the end.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

NAME, VERSION, TRAIN_FRAC = "loco", "none", "0.5"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6", "polars==1.44.2",
                "xgboost==3.2.0"], check=True)
code = os.path.dirname([p for p in glob.glob("/kaggle/input/**/build_candidates.py", recursive=True)
                        if "bedrock-src" in p][0])
data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
tmap = [p for p in glob.glob("/kaggle/input/**/translit_map.parquet", recursive=True) if "bedrock-src" in p][0]
work = "/kaggle/working/artifacts"
for d in ("data", "cands"):
    os.makedirs(f"{work}/{d}", exist_ok=True)
for f in glob.glob(f"{data}/*.parquet"):
    os.symlink(f, f"{work}/data/{os.path.basename(f)}")
WANT = {"train_India", "train_US", "test_France", "test_India", "test_US"}
for f in glob.glob("/kaggle/input/**/*.parquet", recursive=True):
    stem = os.path.splitext(os.path.basename(f))[0]
    if stem in WANT and not os.path.exists(f"{work}/cands/{stem}.parquet"):
        os.symlink(f, f"{work}/cands/{stem}.parquet")
print("cands:", sorted(os.listdir(f"{work}/cands")), flush=True)
assert len(os.listdir(f"{work}/cands")) == 5, "expected 5 candidate files (2 train + 3 test)"
shutil.copy(tmap, f"{work}/translit_map.parquet")
shutil.copytree(code, "/kaggle/working/src", dirs_exist_ok=True)
env = {**os.environ, "BER_ARTIFACT_DIR": work}


def step(*args: str) -> None:
    """Run one pipeline script from /kaggle/working/src and time it."""
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", *args], cwd="/kaggle/working/src", env=env, check=True)
    print(f"=== done in {time.time() - t0:.0f}s", flush=True)


step("features.py", "--split", "train", "--frac", TRAIN_FRAC)
step("extra_features.py", "--split", "train")
step("selftrain.py", "--loco", "--src-frac", "0.6")
for d in ("feats", "feats_extra", "data", "cands"):
    shutil.rmtree(f"{work}/{d}", ignore_errors=True)
