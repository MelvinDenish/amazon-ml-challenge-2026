"""Kaggle GPU job (Bedrock ER): test features + v6 prediction + diagnostics.

Reuses bedrock-k2-gpu outputs (xgb_v6 models, OOF, train feature shards) so
nothing is retrained. Inputs: amazon-ml (data), bedrock-src (code, map),
bedrock-cands (train_India cands), kernel outputs bedrock-blk3-test (test cands),
bedrock-blk3-train-us (train_US cands), bedrock-k2-gpu (models/oof/train feats).
Outputs: submissions/<VERSION>/*.tsv + submission_log.csv, oof/test_<NAME>.parquet,
and the diagnose report in the log.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

NAME, VERSION = "xgb_v6", "v06_pool600_frac50"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6", "polars==1.44.2",
                "xgboost==3.2.0"], check=True)
code = os.path.dirname([p for p in glob.glob("/kaggle/input/**/build_candidates.py", recursive=True)
                        if "bedrock-src" in p][0])
data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
tmap = [p for p in glob.glob("/kaggle/input/**/translit_map.parquet", recursive=True) if "bedrock-src" in p][0]
work = "/kaggle/working/artifacts"
for d in ("data", "cands", "feats", "feats_extra", "models", "oof"):
    os.makedirs(f"{work}/{d}", exist_ok=True)
for f in glob.glob(f"{data}/*.parquet"):
    os.symlink(f, f"{work}/data/{os.path.basename(f)}")
WANT = {"train_India", "train_US", "test_France", "test_India", "test_US"}
for f in glob.glob("/kaggle/input/**/*.parquet", recursive=True):
    stem, parent = os.path.splitext(os.path.basename(f))[0], os.path.basename(os.path.dirname(f))
    if stem in WANT and parent != "feats" and parent != "feats_extra" \
            and not os.path.exists(f"{work}/cands/{stem}.parquet"):
        os.symlink(f, f"{work}/cands/{stem}.parquet")
# v6 models from the private dataset bedrock-v6 (the errored k2 kernel's output is not mountable)
for f in glob.glob("/kaggle/input/**/xgb_v6_fold*.json", recursive=True):
    if not os.path.exists(f"{work}/models/{os.path.basename(f)}"):
        os.symlink(f, f"{work}/models/{os.path.basename(f)}")
print("cands:", sorted(os.listdir(f"{work}/cands")), "| models:", sorted(os.listdir(f"{work}/models")), flush=True)
assert len(os.listdir(f"{work}/cands")) == 5 and any(NAME in m for m in os.listdir(f"{work}/models"))
shutil.copy(tmap, f"{work}/translit_map.parquet")
shutil.copytree(code, "/kaggle/working/src", dirs_exist_ok=True)
env = {**os.environ, "BER_ARTIFACT_DIR": work}


def step(*args: str) -> None:
    """Run one pipeline script from /kaggle/working/src and time it."""
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", *args], cwd="/kaggle/working/src", env=env, check=True)
    print(f"=== done in {time.time() - t0:.0f}s", flush=True)


step("features.py", "--split", "test")
step("extra_features.py", "--split", "test")
step("predict.py", "--name", NAME, "--version", VERSION)
step("diagnose.py", "--name", NAME, "--test-only")
for d in ("feats", "feats_extra", "data", "cands", "models"):
    shutil.rmtree(f"{work}/{d}", ignore_errors=True)
