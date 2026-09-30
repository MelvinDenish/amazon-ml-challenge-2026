"""Kaggle GPU job (Bedrock ER): v07 ensemble = xgb_v6 (reused) + XGBoost variant + CatBoost, OOF blend.

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

NAME, VERSION = "blend_v7", "v07_blend"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6", "polars==1.44.2",
                "xgboost==3.2.0", "catboost"], check=True)
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
for f in glob.glob("/kaggle/input/**/oof/xgb_v6.parquet", recursive=True):
    shutil.copy(f, f"{work}/oof/xgb_v6.parquet")
for f in glob.glob("/kaggle/input/**/xgb_v6_fold*.json", recursive=True):
    if not os.path.exists(f"{work}/models/{os.path.basename(f)}"):
        os.symlink(f, f"{work}/models/{os.path.basename(f)}")
print("cands:", sorted(os.listdir(f"{work}/cands")), "| models:", sorted(os.listdir(f"{work}/models")), flush=True)
assert len(os.listdir(f"{work}/cands")) == 5 and os.path.exists(f"{work}/oof/xgb_v6.parquet") and len(glob.glob(f"{work}/models/xgb_v6_fold*.json")) == 3
shutil.copy(tmap, f"{work}/translit_map.parquet")
shutil.copytree(code, "/kaggle/working/src", dirs_exist_ok=True)
env = {**os.environ, "BER_ARTIFACT_DIR": work}


def step(*args: str) -> None:
    """Run one pipeline script from /kaggle/working/src and time it."""
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", *args], cwd="/kaggle/working/src", env=env, check=True)
    print(f"=== done in {time.time() - t0:.0f}s", flush=True)


step("features.py", "--split", "train", "--frac", "0.5")
step("extra_features.py", "--split", "train")
step("train_gpu.py", "--name", "xgb_v7b", "--folds", "3", "--extra", "--depth", "11", "--colsample", "0.6", "--lr", "0.06", "--seed", "7")
step("ensemble.py", "cat", "--name", "cat_v7")
step("ensemble.py", "blend", "--names", "xgb_v6,xgb_v7b,cat_v7", "--out", NAME)
step("features.py", "--split", "test")
step("extra_features.py", "--split", "test")
import polars as pl  # installed above
os.makedirs(f"{work}/groups", exist_ok=True)
for f in glob.glob(f"{work}/feats_extra/*.parquet"):
    e = pl.read_parquet(f, columns=['s1_id', 'cand_id', 'twin_flag', 'num_absdiff', 'form_disjoint',
                                    'num_prefix', 'cand_num_support', 's1_num_support'])
    b = pl.read_parquet(f.replace('feats_extra', 'feats'), columns=['s1_id', 'cand_id', 't_a_empty', 'n_tset', 'a_tset'])
    e.join(b, on=['s1_id', 'cand_id']).write_parquet(f.replace('feats_extra', 'groups'))
print('groups exported', flush=True)
step("predict.py", "--name", NAME, "--version", VERSION, "--blend")
step("diagnose.py", "--name", NAME)
for d in ("feats", "feats_extra", "data", "cands"):
    shutil.rmtree(f"{work}/{d}", ignore_errors=True)
