"""Second model family (CatBoost) and OOF-weighted blending.

    python ensemble.py cat   --name cat_v7                 # CatBoost, same S1-hash folds, OOF
    python ensemble.py blend --names xgb_v7a,xgb_v7b,cat_v7 --out blend_v7

Why: diverse model families make different errors; a blend chosen on out-of-fold
predictions is a standard, measurable gain. Weights are picked by OOF logloss
over a simplex grid (cheap), then the full-pipeline macro F0.5 is reported for the
chosen blend and for every single model, so the decision uses the real metric.
The blend's weights are saved to models/<out>.json for predict.py --blend.
"""

import argparse
import itertools
import json
import time

import numpy as np
import polars as pl

from config import DEVICE, SEED
from features import ID_COLS
from train import MODEL_DIR, OOF_DIR, evaluate, feature_columns, load_train_features


def train_cat(name: str, folds: int = 3, iters: int = 4000, lr: float = 0.08, depth: int = 8) -> None:
    """CatBoost (GPU when available) with the same folds as train_gpu.py; saves OOF + models."""
    from catboost import CatBoostClassifier
    df = load_train_features(use_extra=True)
    df = df.with_columns((pl.col("s1_id").hash(seed=SEED) % folds).cast(pl.UInt8).alias("fold"))
    feats = feature_columns(df)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = df["fold"].to_numpy()
    ids = df.select(ID_COLS + ["country", "y"])
    del df
    oof = np.zeros(len(y), dtype=np.float32)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(folds):
        t0 = time.time()
        tr, va = fold != k, fold == k
        m = CatBoostClassifier(iterations=iters, learning_rate=lr, depth=depth, loss_function="Logloss",
                               task_type="GPU" if DEVICE == "cuda" else "CPU", random_seed=SEED,
                               od_type="Iter", od_wait=100, verbose=0, border_count=254)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(X[va])[:, 1]
        m.save_model(str(MODEL_DIR / f"{name}_fold{k}.cbm"))
        with open(MODEL_DIR / f"{name}_features.json", "w") as f:
            json.dump(feats, f)
        print(f"  fold {k}: best_iter={m.get_best_iteration()} secs={time.time() - t0:.0f}", flush=True)
    out = ids.with_columns(pl.Series("p", oof))
    out.write_parquet(OOF_DIR / f"{name}.parquet")
    for kk, v in evaluate(out, out["country"].unique().to_list()).items():
        print(kk, v, flush=True)


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy (clipped)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def blend(names: list[str], out: str, step: float = 0.1) -> None:
    """Pick simplex weights by OOF logloss; report full-pipeline F0.5; save blend OOF + weights."""
    base = pl.read_parquet(OOF_DIR / f"{names[0]}.parquet").select(ID_COLS + ["country", "y"])
    for n in names:
        base = base.join(pl.read_parquet(OOF_DIR / f"{n}.parquet").select(ID_COLS + [pl.col("p").alias(n)]),
                         on=ID_COLS)
    y = base["y"].to_numpy().astype(np.float64)
    P = np.column_stack([base[n].to_numpy().astype(np.float64) for n in names])
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, step), repeat=len(names))
            if abs(sum(w) - 1) < 1e-9]
    scored = sorted((_logloss(y, P @ np.array(w)), w) for w in grid)
    best_ll, best_w = scored[0]
    print("single-model logloss:", {n: round(_logloss(y, P[:, i]), 5) for i, n in enumerate(names)})
    print(f"best blend logloss {best_ll:.5f} weights {dict(zip(names, [round(x, 2) for x in best_w]))}", flush=True)
    res = base.select(ID_COLS + ["country", "y"]).with_columns(pl.Series("p", (P @ np.array(best_w)).astype(np.float32)))
    res.write_parquet(OOF_DIR / f"{out}.parquet")
    with open(MODEL_DIR / f"{out}.json", "w") as f:
        json.dump({"models": names, "weights": [float(x) for x in best_w]}, f)
    for n in names:
        single = base.select(ID_COLS + ["country", "y", pl.col(n).alias("p")])
        print(f"[{n}] ef05", evaluate(single, single["country"].unique().to_list())["ef05"], flush=True)
    print(f"[{out}] ef05", evaluate(res, res["country"].unique().to_list())["ef05"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cat", "blend"])
    ap.add_argument("--name", default="cat_v7")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", default="blend_v7")
    a = ap.parse_args()
    if a.mode == "cat":
        train_cat(a.name)
    else:
        blend(a.names.split(","), a.out)
