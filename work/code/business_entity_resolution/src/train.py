"""Train the LightGBM pair classifier with GroupKFold (by S1) and score OOF.

    python train.py --name lgb_v1 [--folds 5]

Outputs
  artifacts/models/<name>_fold{k}.txt   LightGBM boosters
  artifacts/oof/<name>.parquet          OOF probabilities (s1_id, cand_id, p, y)
Prints the full-pipeline macro F0.5 (one-owner + set selection) per country.
"""

import argparse
import time

import numpy as np
import polars as pl

from config import ARTIFACT_DIR, SEED
from features import FEAT_DIR, ID_COLS
from io_utils import load_ground_truth_pairs, load_records
from postprocess import one_owner, select_expected_f05, select_threshold
from score import report

MODEL_DIR = ARTIFACT_DIR / "models"
OOF_DIR = ARTIFACT_DIR / "oof"

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 8,
    "verbose": -1,
    "seed": SEED,
}


def feature_columns(df: pl.DataFrame) -> list[str]:
    """All model features: every column except ids and the label."""
    return [c for c in df.columns if c not in ID_COLS + ["y", "fold", "country"]]


def load_train_features(use_extra: bool = False) -> pl.DataFrame:
    """Load train feature shards, attach label y and country.

    With ``use_extra`` the evidence-driven extra features (extra_features.py:
    legal form, house-number relation, twin competition) are joined on.
    """
    extra_dir = FEAT_DIR.parent / "feats_extra"
    parts = []
    for f in sorted(FEAT_DIR.glob("train_*.parquet")):
        part = pl.read_parquet(f).with_columns(pl.lit(f.stem.split("_", 1)[1]).alias("country"))
        if use_extra:
            part = part.join(pl.read_parquet(extra_dir / f.name), on=ID_COLS, how="left")
        parts.append(part)
    df = pl.concat(parts, how="diagonal_relaxed")
    gt = load_ground_truth_pairs().with_columns(pl.lit(1, dtype=pl.UInt8).alias("y"))
    return df.join(gt, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))


def evaluate(oof: pl.DataFrame, countries: list[str]) -> dict:
    """Full-pipeline metric on the S1s present in ``oof`` (sampled S1 set)."""
    s1 = load_records("train", "source1").filter(pl.col("country").is_in(countries))
    s1 = s1.join(oof.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi")
    truth = load_ground_truth_pairs().join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    owned = one_owner(oof, "p")
    res = {"ef05": report(truth, select_expected_f05(owned, "p"), s1)}
    for thr in (0.4, 0.5, 0.6, 0.7):
        res[f"thr{thr}"] = report(truth, select_threshold(owned, "p", thr), s1)
    return res


def main() -> None:
    """Train fold models, save OOF predictions, and print validation scores."""
    import lightgbm as lgb  # lazy: only this CPU LightGBM path needs it (TPU/GPU images may lack it)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgb_v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=LGB_PARAMS["learning_rate"])
    a = ap.parse_args()
    params = {**LGB_PARAMS, "learning_rate": a.lr}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    df = load_train_features()
    df = df.with_columns((pl.col("s1_id").hash(seed=SEED) % a.folds).cast(pl.UInt8).alias("fold"))
    feats = feature_columns(df)
    print(f"rows={df.height:,} pos_rate={df['y'].mean():.4f} n_feats={len(feats)}", flush=True)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = df["fold"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    importance = np.zeros(len(feats))
    for k in range(a.folds):
        t0 = time.time()
        tr, va = fold != k, fold == k
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        booster = lgb.train(
            params, dtr, num_boost_round=a.rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        importance += booster.feature_importance("gain")
        booster.save_model(str(MODEL_DIR / f"{a.name}_fold{k}.txt"), num_iteration=booster.best_iteration)
        print(f"  fold {k}: best_iter={booster.best_iteration} "
              f"logloss={booster.best_score['valid_0']['binary_logloss']:.5f} secs={time.time() - t0:.0f}", flush=True)
    out = df.select(ID_COLS + ["country", "y"]).with_columns(pl.Series("p", oof))
    out.write_parquet(OOF_DIR / f"{a.name}.parquet")
    top = sorted(zip(importance, feats), reverse=True)[:25]
    print("top features:", [f for _, f in top])
    for k, v in evaluate(out, out["country"].unique().to_list()).items():
        print(k, v, flush=True)


if __name__ == "__main__":
    main()
