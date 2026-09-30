"""GBM stacker: CE + GBM probability + pair context + CE context within the S1 (routed pairs).

Inputs per routed pair: logit p, CE logit, candidate group flags (empty address, twin, number gap,
form conflict, name/address token-set), and CE context inside the S1: rank of this pair's CE among
the S1's routed pairs, max CE of the S1's OTHER routed pairs, number of routed pairs, and the same
for the candidate record across S1s. Cross-fitted 2-fold by S1 on the held-out CE-eval S1s; the
full pipeline is scored on ALL pairs of those S1s (non-routed keep p).

    python ce_stack2.py <base_oof> <ce_eval> [--test <base_test> <ce_test> <out>]
"""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from postprocess import exclusive_owner_prob  # noqa: E402
from train import evaluate  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
GC = ["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty", "n_tset", "a_tset",
      "cand_num_support", "s1_num_support", "num_prefix"]
FEATS = ["lp", "ce", "t_a_empty", "n_tset", "a_tset",
         "ce_rank", "ce_other_max", "n_routed", "ce_cand_other_max", "p_other_max", "s1_psum"]  # no twin/number/form: shift-sensitive
PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 6, "eta": 0.05, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 20, "tree_method": "hist", "device": "cuda", "seed": 42}


def context(d: pl.DataFrame) -> pl.DataFrame:
    """Per-pair features incl. CE context among the S1's routed pairs and the candidate's routed S1s."""
    p = pl.col("p").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    d = d.with_columns(lp=(p / (1 - p)).log(), s1_psum=pl.col("p").sum().over("s1_id"))
    r = d.filter(pl.col("ce").is_not_null())
    r = r.with_columns(
        ce_rank=pl.col("ce").rank("ordinal", descending=True).over("s1_id").cast(pl.Float32),
        n_routed=pl.len().over("s1_id").cast(pl.Float32),
        ce_other_max=pl.col("ce").sort(descending=True).slice(1, 1).first().over("s1_id"),
        ce_cand_other_max=pl.col("ce").sort(descending=True).slice(1, 1).first().over("cand_id"),
    )
    # best other p of the S1 (pairs of any band)
    top2 = d.group_by("s1_id").agg(pl.col("p").sort(descending=True).head(2).alias("tp"))
    r = r.join(top2, on="s1_id", how="left").with_columns(
        p_other_max=pl.when(pl.col("tp").list.first() == pl.col("p")).then(pl.col("tp").list.get(1, null_on_oob=True))
        .otherwise(pl.col("tp").list.first())).drop("tp")
    return r


def load(base: pl.DataFrame, ce: pl.DataFrame, split: str) -> pl.DataFrame:
    g = pl.concat([pl.read_parquet(f"{K5}/groups/{split}_{c}.parquet", columns=GC) for c in
                   (("India", "US") if split == "train" else ("France", "India", "US"))])
    d = base.join(ce, on=["s1_id", "cand_id"], how="left")
    r = context(d).join(g, on=["s1_id", "cand_id"], how="left")
    return d, r


def main() -> None:
    base_path, ce_path = sys.argv[1], sys.argv[2]
    v07_s1 = pl.scan_parquet(f"{K5}/oof/blend_v7.parquet").select("s1_id").unique().collect()
    base = (pl.read_parquet(base_path).join(v07_s1, on="s1_id", how="semi")
            .filter((pl.col("s1_id").hash(seed=45) % 10_000) < 5000))
    d, r = load(base, pl.read_parquet(ce_path), "train")
    X = r.select(FEATS).to_numpy().astype(np.float32)
    y = r["y"].to_numpy()
    fold = (r["s1_id"].hash(seed=99) % 2).to_numpy()
    newp = np.zeros(len(y), dtype=np.float32)
    for k in (0, 1):
        dtr = xgb.DMatrix(X[fold != k], y[fold != k], feature_names=FEATS)
        dva = xgb.DMatrix(X[fold == k], y[fold == k], feature_names=FEATS)
        m = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
        newp[fold == k] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        print(f"fold {k}: best_iter {m.best_iteration} logloss {m.best_score:.5f}", flush=True)
    st = d.join(r.select("s1_id", "cand_id").with_columns(pl.Series("q", newp)), on=["s1_id", "cand_id"], how="left") \
          .with_columns(pl.coalesce("q", "p").alias("p")).drop("q", "ce")
    for c in ("India", "US"):
        x = st.filter(pl.col("country") == c)
        print(f"  GBM-stack owner_prob {c}: {evaluate(exclusive_owner_prob(x, 'p'), [c])['ef05'][c]:.5f}", flush=True)
    if "--test" in sys.argv:
        i = sys.argv.index("--test")
        t_base, t_ce, out = sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3]
        dall = xgb.DMatrix(X, y, feature_names=FEATS)
        full = xgb.train(PARAMS, dall, int(np.mean([1])) or 1)  # placeholder, replaced below
        n_iter = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 800
        full = xgb.train(PARAMS, dall, n_iter)
        tce = pl.read_parquet(t_ce)
        parts = []
        for c in ("France", "India", "US"):
            t = pl.scan_parquet(t_base).filter(pl.col("country") == c).collect()
            if "p" not in t.columns:
                t = t.rename({[x for x in t.columns if x not in ("s1_id", "cand_id", "country")][0]: "p"})
            td, tr_ = load(t.select("s1_id", "cand_id", "country", "p"), tce, "test")
            tr_ = tr_.filter(pl.col("country") == c)
            pr = full.predict(xgb.DMatrix(tr_.select(FEATS).to_numpy().astype(np.float32), feature_names=FEATS))
            td = td.join(tr_.select("s1_id", "cand_id").with_columns(pl.Series("q", pr)), on=["s1_id", "cand_id"], how="left")
            parts.append(td.with_columns(pl.coalesce("q", "p").alias("p")).select("s1_id", "cand_id", "country", "p"))
            print(f"  test {c}: routed {tr_.height:,} mean p {tr_['p'].mean():.4f} -> {pr.mean():.4f}", flush=True)
        pl.concat(parts).write_parquet(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
