"""Train the XGBoost pair classifier on the GPU with GroupKFold (by S1).

    python train_gpu.py --name xgb_v2 [--folds 3] [--device cuda]

Same features, folds and full-pipeline evaluation as train.py, but trained
with XGBoost's CUDA histogram method (a few minutes on an RTX 3050 instead of
hours of CPU LightGBM). Falls back to CPU with --device cpu.

Outputs
  artifacts/models/<name>_fold{k}.json   XGBoost boosters
  artifacts/oof/<name>.parquet           OOF probabilities (s1_id, cand_id, country, y, p)
"""

import argparse
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import DEVICE, SEED
from features import ID_COLS
from train import MODEL_DIR, OOF_DIR, evaluate, feature_columns, load_train_features

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 9,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "max_bin": 256,
    "seed": SEED,
}


def main() -> None:
    """Train fold models on GPU, save OOF predictions, print validation scores."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xgb_v2")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--extra", action="store_true", help="join extra_features.py shards")
    ap.add_argument("--train-subsample", type=float, default=1.0,
                    help="learning curve: train each fold on this share of its training S1 "
                         "(validation S1s unchanged, so OOF scores are directly comparable)")
    ap.add_argument("--depth", type=int, default=XGB_PARAMS["max_depth"], help="ensemble variant: tree depth")
    ap.add_argument("--colsample", type=float, default=XGB_PARAMS["colsample_bytree"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude", default="", help="ablation: comma-separated feature columns to drop "
                    "(same rows and folds, so the OOF difference isolates those features)")
    a = ap.parse_args()
    params = {**XGB_PARAMS, "eta": a.lr, "device": a.device, "max_depth": a.depth,
              "colsample_bytree": a.colsample, "seed": a.seed}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train_features(use_extra=a.extra)
    df = df.with_columns((pl.col("s1_id").hash(seed=SEED) % a.folds).cast(pl.UInt8).alias("fold"))
    drop = {f for f in a.exclude.split(",") if f}
    feats = [f for f in feature_columns(df) if f not in drop]
    assert len(drop & set(feature_columns(df))) == len(drop), f"unknown --exclude columns: {drop - set(feature_columns(df))}"
    print(f"rows={df.height:,} pos_rate={df['y'].mean():.4f} n_feats={len(feats)} device={a.device}", flush=True)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = df["fold"].to_numpy()
    keep = (df["s1_id"].hash(seed=SEED + 1) % 10_000).to_numpy() < a.train_subsample * 10_000
    oof = np.zeros(len(y), dtype=np.float32)
    gain: dict[str, float] = {}
    for k in range(a.folds):
        t0 = time.time()
        tr, va = (fold != k) & keep, fold == k
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], feature_names=feats, max_bin=XGB_PARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(X[va], y[va], feature_names=feats, ref=dtr)
        booster = xgb.train(
            params, dtr, num_boost_round=a.rounds, evals=[(dva, "valid")],
            early_stopping_rounds=100, verbose_eval=False,
        )
        oof[va] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        for f, g in booster.get_score(importance_type="gain").items():
            gain[f] = gain.get(f, 0.0) + g
        # Save only the early-stopped trees so test inference matches the OOF scoring
        # (unsliced models kept 100 extra rounds that OOF never used).
        booster[: booster.best_iteration + 1].save_model(str(MODEL_DIR / f"{a.name}_fold{k}.json"))
        print(f"  fold {k}: best_iter={booster.best_iteration} logloss={booster.best_score:.5f} "
              f"secs={time.time() - t0:.0f}", flush=True)
        del dtr, dva
    out = df.select(ID_COLS + ["country", "y"]).with_columns(pl.Series("p", oof))
    out.write_parquet(OOF_DIR / f"{a.name}.parquet")
    print("top features:", sorted(gain, key=gain.get, reverse=True)[:25])
    for k, v in evaluate(out, out["country"].unique().to_list()).items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
