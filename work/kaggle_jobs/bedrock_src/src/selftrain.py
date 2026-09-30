"""Self-training (pseudo-labelling) on an unlabelled target domain.

    python selftrain.py --loco        # EXPERIMENT: does it help? (US = source, India = target)

Why: the leaderboard gap (OOF 0.975 vs LB 0.962) is train->test shift, and 15%
of the test (France) has no labels at all. Self-training adds the model's most
confident predictions on the *unlabelled* target as training rows, so the model
adapts to the target's feature distribution (e.g. its twins and naming style).

The LOCO experiment measures this with real labels before it is ever used:
  A  train on US only                      -> score India (full pipeline F0.5)
  B  train on US + confident India pseudo-labels (p>=HI -> 1, p<=LO -> 0)
                                           -> score India with the retrained model
India labels are used ONLY for scoring, never for training. B - A is the
measured value of self-training under a country shift.
"""

import argparse
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import DEVICE, SEED
from features import ID_COLS
from io_utils import load_ground_truth_pairs, load_records
from postprocess import one_owner, select_expected_f05
from score import report
from train import feature_columns, load_train_features
from train_gpu import XGB_PARAMS

HI, LO = 0.97, 0.03


def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, feats: list[str], holdout: np.ndarray) -> xgb.Booster:
    """GPU XGBoost with early stopping on a source-domain holdout."""
    params = {**XGB_PARAMS, "eta": 0.08, "device": DEVICE}
    dtr = xgb.QuantileDMatrix(X[~holdout], y[~holdout], weight=w[~holdout], feature_names=feats,
                              max_bin=XGB_PARAMS["max_bin"])
    dva = xgb.QuantileDMatrix(X[holdout], y[holdout], feature_names=feats, ref=dtr)
    return xgb.train(params, dtr, 3000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=False)


def predict(b: xgb.Booster, X: np.ndarray, feats: list[str]) -> np.ndarray:
    """Chunked GPU prediction (bounded memory)."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), 2_000_000):
        out[s:s + 2_000_000] = b.predict(xgb.DMatrix(X[s:s + 2_000_000], feature_names=feats),
                                         iteration_range=(0, b.best_iteration + 1))
    return out


def score_target(pairs: pl.DataFrame, p: np.ndarray, country: str) -> dict:
    """Full-pipeline macro F0.5 on the target S1s, using their real labels (scoring only)."""
    pr = pairs.select(ID_COLS).with_columns(pl.Series("p", p))
    s1 = load_records("train", "source1").join(pr.select(pl.col("s1_id").alias("entity_id")).unique(),
                                               on="entity_id", how="semi")
    gt = load_ground_truth_pairs().join(pr.select("s1_id").unique(), on="s1_id", how="semi")
    return report(gt, select_expected_f05(one_owner(pr, "p"), "p"), s1)[country]


def loco(source: str = "US", target: str = "India", src_frac: float = 1.0) -> None:
    """Run experiment A (source only) and B (source + target pseudo-labels)."""
    df = load_train_features(use_extra=True)
    feats = feature_columns(df)
    src = df.filter(pl.col("country") == source)
    tgt = df.filter(pl.col("country") == target)
    del df
    if src_frac < 1.0:  # same subset for A and B, so the comparison stays fair
        src = src.filter((pl.col("s1_id").hash(seed=SEED + 2) % 10_000) < src_frac * 10_000)
    Xs, ys = src.select(feats).to_numpy().astype(np.float32), src["y"].to_numpy().astype(np.float32)
    hold = ((src["s1_id"].hash(seed=SEED) % 10) == 0).to_numpy()
    del src  # keep only numpy arrays for the source (memory: the Kaggle run OOM'd)
    Xt = tgt.select(feats).to_numpy().astype(np.float32)
    tgt = tgt.select(ID_COLS + ["y"])

    t0 = time.time()
    a = fit(Xs, ys, np.ones(len(ys), np.float32), feats, hold)
    pa = predict(a, Xt, feats)
    fa = score_target(tgt, pa, target)
    print(f"A  train {source} only -> {target} F0.5 = {fa:.4f}  (iters {a.best_iteration}, {time.time() - t0:.0f}s)", flush=True)

    conf = (pa >= HI) | (pa <= LO)
    yp = (pa >= HI).astype(np.float32)
    print(f"   pseudo-labels: {conf.mean():.3f} of {target} pairs confident "
          f"(pos {int((pa >= HI).sum()):,}, neg {int((pa <= LO).sum()):,}); "
          f"pseudo-label precision vs hidden truth: pos {tgt['y'].to_numpy()[pa >= HI].mean():.4f} "
          f"neg {1 - tgt['y'].to_numpy()[pa <= LO].mean():.4f}", flush=True)
    X2 = np.vstack([Xs, Xt[conf]])
    del Xs
    y2 = np.r_[ys, yp[conf]]
    w2 = np.ones(len(y2), np.float32)
    h2 = np.r_[hold, np.zeros(int(conf.sum()), bool)]
    del a
    t0 = time.time()
    b = fit(X2, y2, w2, feats, h2)
    pb = predict(b, Xt, feats)
    fb = score_target(tgt, pb, target)
    print(f"B  train {source} + {target} pseudo-labels -> {target} F0.5 = {fb:.4f}  "
          f"(delta {fb - fa:+.4f}, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", action="store_true")
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    ap.add_argument("--src-frac", type=float, default=1.0)
    a = ap.parse_args()
    if a.loco:
        loco(a.source, a.target, a.src_frac)
