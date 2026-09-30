"""Kaggle GPU job (Bedrock ER): v08 = v2 raw-address / name-difference features, with a same-sample ablation.

Everything is trained on ONE deterministic 50% S1 sample (hash rule in features.py),
so xgb_v8 vs xgb_v8_abl (identical rows/folds, v2 columns dropped) isolates the
v2 feature effect. Models are sliced to their early-stopped trees.
Order puts the must-have outputs first (xgb_v8 submission + diagnose); CatBoost and
the blend run last and may fail without losing the earlier outputs.
Inputs: amazon-ml (data), bedrock-src (code, map), bedrock-cands (train_India cands),
kernel outputs bedrock-blk3-test (test cands), bedrock-blk3-train-us (train_US cands).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

V2 = ("numtok_jacc,numtok_q_only,numtok_t_only,compound_shared,compound_conflict,"
      "unit_eq,unit_conflict,name_q_only,name_t_only")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6", "polars==1.44.2",
                "xgboost==3.2.0", "catboost"], check=True)
code = os.path.dirname([p for p in glob.glob("/kaggle/input/**/build_candidates.py", recursive=True)
                        if "bedrock-src" in p][0])
data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
tmap = [p for p in glob.glob("/kaggle/input/**/translit_map.parquet", recursive=True) if "bedrock-src" in p][0]
work = "/kaggle/working/artifacts"
for d in ("data", "cands", "feats", "feats_extra", "models", "oof", "groups"):
    os.makedirs(f"{work}/{d}", exist_ok=True)
for f in glob.glob(f"{data}/*.parquet"):
    os.symlink(f, f"{work}/data/{os.path.basename(f)}")
WANT = {"train_India", "train_US", "test_France", "test_India", "test_US"}
for f in glob.glob("/kaggle/input/**/*.parquet", recursive=True):
    stem, parent = os.path.splitext(os.path.basename(f))[0], os.path.basename(os.path.dirname(f))
    if stem in WANT and parent not in ("feats", "feats_extra") and not os.path.exists(f"{work}/cands/{stem}.parquet"):
        os.symlink(f, f"{work}/cands/{stem}.parquet")
print("cands:", sorted(os.listdir(f"{work}/cands")), flush=True)
assert len(os.listdir(f"{work}/cands")) == 5
src_text = open(f"{code}/extra_features.py", encoding="utf-8").read()
assert '"pool_k": 600' in open(f"{code}/build_candidates.py", encoding="utf-8").read()
assert "numtok_jacc" in src_text and "zip_eq" not in src_text, "stale bedrock-src version"
assert "exclusive_owner_prob" in open(f"{code}/predict.py", encoding="utf-8").read()
assert "hash(seed=SEED + 3)" in open(f"{code}/features.py", encoding="utf-8").read()
shutil.copy(tmap, f"{work}/translit_map.parquet")
shutil.copytree(code, "/kaggle/working/src", dirs_exist_ok=True)
env = {**os.environ, "BER_ARTIFACT_DIR": work}


def step(*args: str, check: bool = True) -> bool:
    """Run one pipeline script from /kaggle/working/src and time it."""
    t0 = time.time()
    print(f"\n=== {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", *args], cwd="/kaggle/working/src", env=env)
    print(f"=== exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    if check and r.returncode:
        raise SystemExit(f"step failed: {args}")
    return r.returncode == 0


CHECK = f"""
import glob, polars as pl
from io_utils import load_ground_truth_pairs
V2 = "{V2}".split(",") + ["twin_flag", "form_disjoint"]
split = "SPLIT"
gt = load_ground_truth_pairs().with_columns(pl.lit(1, dtype=pl.UInt8).alias("y")) if split == "train" else None
for f in sorted(glob.glob("{work}/feats_extra/" + split + "_*.parquet")):
    x = pl.read_parquet(f, columns=["s1_id", "cand_id"] + V2)
    keys = []
    if gt is not None:
        x = x.join(gt, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0)); keys = ["y"]
    agg = [pl.len().alias("n")] + [pl.col(c).cast(pl.Float64).mean().round(4).alias(c) for c in V2]
    print(f.rsplit("/", 1)[1], (x.group_by(keys).agg(agg).sort(keys) if keys else x.select(agg)).to_dicts(), flush=True)
"""

step("features.py", "--split", "train", "--frac", "0.5")
step("extra_features.py", "--split", "train")
step("-c", CHECK.replace("SPLIT", "train"), check=False)
step("train_gpu.py", "--name", "xgb_v8", "--folds", "3", "--extra")
step("train_gpu.py", "--name", "xgb_v8_abl", "--folds", "3", "--extra", "--exclude", V2)
step("features.py", "--split", "test")
step("extra_features.py", "--split", "test")
step("-c", CHECK.replace("SPLIT", "test"), check=False)
import polars as pl  # installed above
for f in glob.glob(f"{work}/feats_extra/*.parquet"):
    e = pl.read_parquet(f, columns=["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint",
                                    "num_prefix", "cand_num_support", "s1_num_support"])
    b = pl.read_parquet(f.replace("feats_extra", "feats"), columns=["s1_id", "cand_id", "t_a_empty", "n_tset", "a_tset"])
    e.join(b, on=["s1_id", "cand_id"]).write_parquet(f.replace("feats_extra", "groups"))
print("groups exported", flush=True)
step("predict.py", "--name", "xgb_v8", "--version", "v08_xgb")
step("predict.py", "--name", "xgb_v8", "--version", "v08_xgb_own", "--reuse", "--owner-prob", check=False)
step("predict.py", "--name", "xgb_v8_abl", "--version", "v08_abl", check=False)
step("diagnose.py", "--name", "xgb_v8", "--name", "xgb_v8_abl", check=False)
# optional tail: second model family + blend (a failure here keeps everything above)
if step("ensemble.py", "cat", "--name", "cat_v8", check=False):
    if step("ensemble.py", "blend", "--names", "xgb_v8,cat_v8", "--out", "blend_v8", check=False):
        step("predict.py", "--name", "blend_v8", "--version", "v08_blend", "--blend", check=False)
        step("predict.py", "--name", "blend_v8", "--version", "v08_blend_own", "--reuse", "--owner-prob", check=False)
        step("diagnose.py", "--name", "blend_v8", "--test-only", check=False)
for d in ("feats", "feats_extra", "data", "cands"):
    shutil.rmtree(f"{work}/{d}", ignore_errors=True)
