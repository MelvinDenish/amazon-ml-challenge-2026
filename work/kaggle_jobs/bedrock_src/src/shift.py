"""Test-like local validation via covariate-shift (importance) weighting.

    python shift.py --oof xgb_v3            # proxy leaderboard score for a model's OOF

Why: the test set differs from train (23% more S2/S3 records per S1, 1.6-3x more
same-name/same-street/different-number "twin" candidates), so plain OOF F0.5
over-estimates the leaderboard (v03: OOF 0.969 vs LB 0.9497).

How (label-free, identical code for train and test):
  1. S1 descriptors aggregated from the candidate-pair feature shards
     (candidate counts, near-identical same-number members, twins, empty-address
     same-name candidates, native-script share, similarity/score statistics).
  2. Adversarial classifier (XGBoost, GPU) per test country: train S1 vs test S1
     (France: all train S1 vs test France). Out-of-fold probabilities give an
     importance weight w = p / (1 - p) for every train S1, normalised to mean 1.
  3. Proxy = sum over test countries of share(country) * weighted mean of the
     per-S1 F0.5 of the model's OOF predictions (full pipeline: one-owner +
     expected-F0.5 selection).
The proxy is only trusted after it reproduces a real leaderboard score.
"""

import argparse

import numpy as np
import polars as pl
import xgboost as xgb

from config import ARTIFACT_DIR, SEED
from features import FEAT_DIR
from io_utils import load_ground_truth_pairs, load_records
from postprocess import one_owner, select_expected_f05
from score import per_entity_f05

SHIFT_DIR = ARTIFACT_DIR / "shift"


def s1_descriptors(split: str, country: str) -> pl.DataFrame:
    """Aggregate label-free per-S1 descriptors from one feature shard."""
    f = pl.read_parquet(FEAT_DIR / f"{split}_{country}.parquet")
    hi_name = pl.col("n_tset") >= 0.9
    return f.group_by("s1_id").agg(
        n_cands=pl.len(),
        n_member=((pl.col("num_eq") == 1) & hi_name).sum(),
        n_twin=((pl.col("num_conflict") == 1) & hi_name & (pl.col("a_tset") >= 0.6)).sum(),
        n_empty_same=((pl.col("t_a_empty") == 1) & hi_name).sum(),
        n_hi_name=hi_name.sum(),
        n_hi_addr=(pl.col("a_tset") >= 0.9).sum(),
        n_native=pl.col("t_was_native").sum(),
        n_domain=pl.col("t_is_domain").sum(),
        n_alias=pl.col("t_has_alias").sum(),
        n_strong=(pl.col("rscore") >= 1.7).sum(),
        rscore_max=pl.col("rscore").max(),
        rscore_mean=pl.col("rscore").mean(),
        bscore_max=pl.col("bscore").max(),
        n_s1_cand_mean=pl.col("n_s1_cand").mean(),
        q_name_cnt=pl.col("q_n_core_cnt").first(),
        q_addr_cnt=pl.col("q_a_norm_cnt").first(),
        q_ntok=pl.col("q_n_ntok").first(),
    ).with_columns(pl.all().exclude("s1_id").cast(pl.Float32))


def adversarial_weights(train_d: pl.DataFrame, test_d: pl.DataFrame, folds: int = 3) -> tuple[np.ndarray, float, list]:
    """Out-of-fold P(test | descriptors) for train rows -> importance weights (mean 1).

    Returns (weights for train rows, adversarial AUC, top shifted features).
    """
    cols = [c for c in train_d.columns if c != "s1_id"]
    X = np.vstack([train_d.select(cols).to_numpy(), test_d.select(cols).to_numpy()]).astype(np.float32)
    y = np.r_[np.zeros(train_d.height), np.ones(test_d.height)]
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, folds, len(y))
    p = np.zeros(len(y))
    params = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
              "device": "cuda", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "seed": SEED}
    gain: dict[str, float] = {}
    for k in range(folds):
        tr, va = fold != k, fold == k
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], feature_names=cols)
        dva = xgb.QuantileDMatrix(X[va], y[va], feature_names=cols, ref=dtr)
        b = xgb.train(params, dtr, 400, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
        p[va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        for f, g in b.get_score(importance_type="gain").items():
            gain[f] = gain.get(f, 0.0) + g
    # AUC via rank statistic
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    pt = np.clip(p[: train_d.height], 1e-4, 1 - 1e-4)
    w = np.clip(pt / (1 - pt), 0.0, None)
    w = np.clip(w / w.mean(), 0.05, 20.0)
    w = w / w.mean()
    top = sorted(gain, key=gain.get, reverse=True)[:6]
    return w, float(auc), top


def build_weights() -> pl.DataFrame:
    """Importance weights for every sampled train S1, one column per test country."""
    tr = {c: s1_descriptors("train", c) for c in ("India", "US")}
    out = []
    for c in ("India", "US"):
        w, auc, top = adversarial_weights(tr[c], s1_descriptors("test", c))
        ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
        print(f"[{c}] adversarial AUC={auc:.3f}  ESS={ess:.2f}  most shifted: {top}", flush=True)
        out.append(tr[c].select("s1_id").with_columns(pl.lit(c).alias("country"), pl.Series("w_self", w)))
    all_tr = pl.concat([tr["India"], tr["US"]])
    w, auc, top = adversarial_weights(all_tr, s1_descriptors("test", "France"))
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    print(f"[France vs all train] adversarial AUC={auc:.3f}  ESS={ess:.2f}  most shifted: {top}", flush=True)
    wf = all_tr.select("s1_id").with_columns(pl.Series("w_france", w))
    res = pl.concat(out).join(wf, on="s1_id")
    SHIFT_DIR.mkdir(parents=True, exist_ok=True)
    res.write_parquet(SHIFT_DIR / "s1_weights.parquet")
    return res


def test_shares() -> dict[str, float]:
    """Share of each country among test S1 (the leaderboard's macro-average mix)."""
    s = load_records("test", "source1").group_by("country").len()
    tot = s["len"].sum()
    return {r["country"]: r["len"] / tot for r in s.to_dicts()}


def proxy_score(oof_name: str, weights: pl.DataFrame | None = None) -> dict:
    """Unweighted and shift-weighted full-pipeline F0.5 of a model's OOF predictions."""
    if weights is None:
        weights = pl.read_parquet(SHIFT_DIR / "s1_weights.parquet")
    oof = pl.read_parquet(ARTIFACT_DIR / "oof" / f"{oof_name}.parquet")
    ids = oof["s1_id"].unique()
    gt = load_ground_truth_pairs().join(oof.select("s1_id").unique(), on="s1_id", how="semi")
    f = per_entity_f05(gt, select_expected_f05(one_owner(oof, "p"), "p"), ids).join(weights, on="s1_id")
    res = {}
    for c in ("India", "US"):
        fc = f.filter(pl.col("country") == c)
        res[f"{c}_plain"] = float(fc["f05"].mean())
        res[f"{c}_shift"] = float((fc["f05"] * fc["w_self"]).sum() / fc["w_self"].sum())
    res["France_shift"] = float((f["f05"] * f["w_france"]).sum() / f["w_france"].sum())
    sh = test_shares()
    res["proxy_LB"] = sh["US"] * res["US_shift"] + sh["India"] * res["India_shift"] + sh["France"] * res["France_shift"]
    res["plain_testmix"] = (sh["US"] * res["US_plain"] + sh["India"] * res["India_plain"]
                            + sh["France"] * min(res["US_plain"], res["India_plain"]))
    return {k: round(v, 4) for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="xgb_v3")
    ap.add_argument("--rebuild-weights", action="store_true")
    a = ap.parse_args()
    wts = build_weights() if a.rebuild_weights or not (SHIFT_DIR / "s1_weights.parquet").exists() else None
    print(a.oof, proxy_score(a.oof, wts))
