"""Stack one or more cross-encoders onto the GBM probability (production + evaluation).

    python ce_stack.py --base-oof blend_v7 --base-test test_blend_v7 \
        --ce distil=ce_eval_scores.parquet:ce_test_scores.parquet \
        --ce xlmr=ce_eval_scores_xlmr.parquet:ce_test_scores_xlmr.parquet --out test_blend_v7_ce

Routing: a pair is re-scored iff it has CE scores (the GBM's uncertainty band, identical on the
held-out evaluation S1s and on test). The stacker is a small XGBoost on: logit p, every CE logit,
empty-address flag, name/address token-set similarity, and CE context inside the S1 (rank of the
pair, best OTHER routed pair, number of routed pairs, best other S1 for the same record, best other
p of the S1, sum of p). Twin / house-number / legal-form features are deliberately left out: their
train-calibrated effect is exactly what shifts on test (they raised the test-side twin counts).
Cross-fitted 2-fold by S1 on the evaluation S1s; the full pipeline (exclusive-owner probabilities,
one-owner, expected-F0.5) is scored on ALL pairs of those S1s. Then refit on all evaluation pairs
and applied to test; writes oof/<out>.parquet (test) and oof/<out>_evaloof.parquet.
"""

import argparse

import numpy as np
import polars as pl
import xgboost as xgb

from config import ARTIFACT_DIR, SEED
from postprocess import exclusive_owner_prob
from train import evaluate

OOF_DIR = ARTIFACT_DIR / "oof"
CE_DIR = ARTIFACT_DIR / "ce"
GROUPS_DIR = ARTIFACT_DIR / "groups"
SRC_FEATS = True  # S1-level source structure: 85% of true clusters have records in BOTH S2 and S3
GC = ["s1_id", "cand_id", "t_a_empty", "n_tset", "a_tset"]
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 6, "eta": 0.05, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 20, "tree_method": "hist", "seed": SEED}


def features(d: pl.DataFrame, ce_names: list[str], split: str) -> pl.DataFrame:
    """Routed pairs with stacker inputs (d holds ALL pairs of the S1s, CE columns null when not routed)."""
    p = pl.col("p").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    is3 = pl.col("cand_id").str.starts_with("S3-")
    conf = (pl.col("p") >= 0.9).cast(pl.Int32)
    d = d.with_columns(lp=(p / (1 - p)).log(), s1_psum=pl.col("p").sum().over("s1_id"),
                       ce_mean=pl.mean_horizontal(ce_names),
                       n_conf_s2=(conf * (~is3).cast(pl.Int32)).sum().over("s1_id"),
                       n_conf_s3=(conf * is3.cast(pl.Int32)).sum().over("s1_id"))
    d = d.with_columns(same_src_conf=pl.when(is3).then(pl.col("n_conf_s3")).otherwise(pl.col("n_conf_s2")).cast(pl.Float32),
                       other_src_conf=pl.when(is3).then(pl.col("n_conf_s2")).otherwise(pl.col("n_conf_s3")).cast(pl.Float32))
    top2 = d.group_by("s1_id").agg(pl.col("p").sort(descending=True).head(2).alias("tp"))
    r = d.filter(pl.col(ce_names[0]).is_not_null()).join(top2, on="s1_id", how="left").with_columns(
        p_other_max=pl.when(pl.col("tp").list.first() == pl.col("p")).then(pl.col("tp").list.get(1, null_on_oob=True))
        .otherwise(pl.col("tp").list.first()),
        ce_rank=pl.col("ce_mean").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
        n_routed=pl.len().over("s1_id").cast(pl.Float32),
        ce_other_max=pl.col("ce_mean").sort(descending=True).slice(1, 1).first().over("s1_id"),
        ce_cand_other_max=pl.col("ce_mean").sort(descending=True).slice(1, 1).first().over("cand_id"),
    ).drop("tp")
    g = pl.concat([pl.read_parquet(f, columns=GC) for f in sorted(GROUPS_DIR.glob(f"{split}_*.parquet"))])
    return r.join(g, on=["s1_id", "cand_id"], how="left")


def feat_names(ce_names: list[str]) -> list[str]:
    """Stacker input columns (order matters for the booster)."""
    return ["lp", *ce_names, "t_a_empty", "n_tset", "a_tset", "ce_rank", "ce_other_max", "n_routed",
            "ce_cand_other_max", "p_other_max", "s1_psum"] + (["same_src_conf", "other_src_conf"] if SRC_FEATS else [])


def attach_ce(base: pl.DataFrame, ce: dict[str, str]) -> pl.DataFrame:
    """Left-join every CE score file as a column named after the CE."""
    for name, path in ce.items():
        base = base.join(pl.read_parquet(CE_DIR / path).rename({"ce": name}), on=["s1_id", "cand_id"], how="left")
    return base


def main() -> None:
    """Cross-fit and score the stacker on the eval S1s, then refit and apply to test."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-oof", required=True)
    ap.add_argument("--base-test", required=True)
    ap.add_argument("--ce", action="append", required=True, help="name=eval_scores.parquet:test_scores.parquet")
    ap.add_argument("--eval-s1-from", default="blend_v7", help="OOF whose S1s (within the v08 hash sample) form the eval set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth", type=int, default=PARAMS["max_depth"])
    ap.add_argument("--eta", type=float, default=PARAMS["eta"])
    ap.add_argument("--mcw", type=float, default=PARAMS["min_child_weight"])
    ap.add_argument("--no-src-feats", action="store_true", help="drop same_src_conf/other_src_conf (the v09 France stack)")
    a = ap.parse_args()
    global SRC_FEATS
    SRC_FEATS = not a.no_src_feats
    params = {**PARAMS, "device": a.device, "max_depth": a.depth, "eta": a.eta, "min_child_weight": a.mcw}
    ce_eval = {k: v.split(":")[0] for k, v in (x.split("=") for x in a.ce)}
    ce_test = {k: v.split(":")[1] for k, v in (x.split("=") for x in a.ce)}
    names = list(ce_eval)
    fn = feat_names(names)

    s1 = pl.scan_parquet(OOF_DIR / f"{a.eval_s1_from}.parquet").select("s1_id").unique().collect()
    base = (pl.read_parquet(OOF_DIR / f"{a.base_oof}.parquet").join(s1, on="s1_id", how="semi")
            .filter((pl.col("s1_id").hash(seed=SEED + 3) % 10_000) < 5000))
    d = attach_ce(base, ce_eval)
    r = features(d, names, "train")
    X, y = r.select(fn).to_numpy().astype(np.float32), r["y"].to_numpy()
    fold = (r["s1_id"].hash(seed=99) % 2).to_numpy()
    newp, iters = np.zeros(len(y), dtype=np.float32), []
    for k in (0, 1):
        dtr = xgb.DMatrix(X[fold != k], y[fold != k], feature_names=fn)
        dva = xgb.DMatrix(X[fold == k], y[fold == k], feature_names=fn)
        m = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
        newp[fold == k] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        iters.append(m.best_iteration + 1)
        print(f"fold {k}: best_iter {m.best_iteration} logloss {m.best_score:.5f}", flush=True)
    st = (d.join(r.select("s1_id", "cand_id").with_columns(pl.Series("q", newp)), on=["s1_id", "cand_id"], how="left")
          .with_columns(pl.coalesce("q", "p").alias("p")).select("s1_id", "cand_id", "country", "y", "p"))
    st.write_parquet(OOF_DIR / f"{a.out}_evaloof.parquet")
    for tag, df in (("base", d.select("s1_id", "cand_id", "country", "y", "p")), ("stacked", st)):
        res = {c: evaluate(exclusive_owner_prob(df.filter(pl.col("country") == c), "p"), [c])["ef05"][c]
               for c in ("India", "US")}
        print(f"  {tag:8s} (owner prob) India {res['India']:.5f} US {res['US']:.5f}", flush=True)

    full = xgb.train(params, xgb.DMatrix(X, y, feature_names=fn), int(np.mean(iters) * 1.2))
    parts = []
    for c in ("France", "India", "US"):
        t = pl.scan_parquet(OOF_DIR / f"{a.base_test}.parquet").filter(pl.col("country") == c).collect()
        if "p" not in t.columns:
            t = t.rename({[x for x in t.columns if x not in ("s1_id", "cand_id", "country")][0]: "p"})
        td = attach_ce(t.select("s1_id", "cand_id", "country", "p"), ce_test)
        tr_ = features(td, names, "test")
        pr = full.predict(xgb.DMatrix(tr_.select(fn).to_numpy().astype(np.float32), feature_names=fn))
        td = td.join(tr_.select("s1_id", "cand_id").with_columns(pl.Series("q", pr)), on=["s1_id", "cand_id"], how="left")
        parts.append(td.with_columns(pl.coalesce("q", "p").alias("p")).select("s1_id", "cand_id", "country", "p"))
        print(f"  test {c}: routed {tr_.height:,} mean p {tr_['p'].mean():.4f} -> {pr.mean():.4f}", flush=True)
    pl.concat(parts).write_parquet(OOF_DIR / f"{a.out}.parquet")
    print("wrote", OOF_DIR / f"{a.out}.parquet")


if __name__ == "__main__":
    main()
