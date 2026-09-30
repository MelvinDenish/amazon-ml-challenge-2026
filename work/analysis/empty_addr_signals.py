"""Is there learnable signal left for EMPTY-ADDRESS candidates after the CE stack?

On the held-out eval S1s (stacked OOF), take candidate records whose address is empty and build
cheap cluster/competition signals the stack does not have:
  src_same_conf   the S1 already has a confident (p>=0.9) record from the SAME source (S2/S3)
  n_conf          number of confident records of the S1
  exact_s1        folded name == S1 folded name
  exact_member    folded name == folded name of one of the S1's confident members
  dup_empty       how many empty-address candidates of this S1 share this folded name
  comp_s1         how many OTHER S1s have this record as a candidate with p >= 0.05
  best_other_p    best p of this record for another S1
Then a 2-fold cross-fitted GBM on empty-address pairs (stack p + signals) vs stack p alone:
logloss and full-pipeline macro F0.5 on all pairs of the eval S1s.

    python empty_addr_signals.py <stacked_evaloof.parquet>
"""
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402
from postprocess import exclusive_owner_prob  # noqa: E402
from train import evaluate  # noqa: E402

FOLD = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]", ""))
EMPTY = pl.col("business_address").fill_null("").str.strip_chars().str.to_lowercase().is_in(["", "null", "<null>", "none"])
FEATS = ["lp", "src_same_conf", "n_conf", "exact_s1", "exact_member", "dup_empty", "comp_s1", "best_other_p", "is_s3"]


def main() -> None:
    oof = pl.read_parquet(sys.argv[1])
    for c in ("India", "US"):
        o = oof.filter(pl.col("country") == c)
        rec = pl.concat([load_records("train", s, c) for s in ("source1", "source2", "source3")]).select(
            "entity_id", FOLD.alias("fn"), EMPTY.alias("empty"))
        d = (o.join(rec.select(pl.col("entity_id").alias("s1_id"), pl.col("fn").alias("s1_fn")), on="s1_id")
             .join(rec.rename({"entity_id": "cand_id"}), on="cand_id")
             .with_columns(is_s3=pl.col("cand_id").str.starts_with("S3-").cast(pl.Int8)))
        conf = d.filter(pl.col("p") >= 0.9)
        d = d.with_columns(n_conf=(pl.col("p") >= 0.9).sum().over("s1_id"),
                           exact_s1=(pl.col("fn") == pl.col("s1_fn")).cast(pl.Int8))
        src = conf.group_by("s1_id").agg(pl.col("is_s3").max().alias("has_s3"), (1 - pl.col("is_s3")).max().alias("has_s2"))
        mem = conf.select("s1_id", pl.col("fn").alias("mfn")).unique().with_columns(pl.lit(1, dtype=pl.Int8).alias("exact_member"))
        e = d.filter(pl.col("empty"))
        e = (e.join(src, on="s1_id", how="left")
             .with_columns(src_same_conf=pl.when(pl.col("is_s3") == 1).then(pl.col("has_s3")).otherwise(pl.col("has_s2")).fill_null(0))
             .drop("has_s3", "has_s2")
             .join(mem, left_on=["s1_id", "fn"], right_on=["s1_id", "mfn"], how="left")
             .with_columns(pl.col("exact_member").fill_null(0), dup_empty=pl.len().over("s1_id", "fn")))
        comp = o.filter(pl.col("p") >= 0.05).group_by("cand_id").agg(pl.len().alias("n_s1"))
        top = o.group_by("cand_id").agg(pl.col("p").sort(descending=True).head(2).alias("tp"))
        pc = pl.col("p").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
        e = (e.join(comp, on="cand_id", how="left").join(top, on="cand_id", how="left")
             .with_columns(comp_s1=(pl.col("n_s1").fill_null(0) - (pl.col("p") >= 0.05).cast(pl.UInt32)).cast(pl.Int32),
                           best_other_p=pl.when(pl.col("tp").list.first() == pl.col("p")).then(pl.col("tp").list.get(1, null_on_oob=True))
                           .otherwise(pl.col("tp").list.first()),
                           lp=(pc / (1 - pc)).log())
             .drop("n_s1", "tp"))
        band = e.filter((pl.col("p") > 0.005) & (pl.col("p") < 0.995))
        print(f"[{c}] empty-address pairs {e.height:,}, in band {band.height:,}, true rate in band {band['y'].mean():.3f}")
        for f in ("src_same_conf", "exact_member", "exact_s1"):
            print(f"   true rate by {f}:", band.group_by(f).agg(pl.len(), pl.col("y").mean().round(3)).sort(f).to_dicts())
        print("   true rate by dup_empty (1/2/3+):", band.with_columns(pl.col("dup_empty").clip(upper_bound=3)).group_by("dup_empty")
              .agg(pl.len(), pl.col("y").mean().round(3)).sort("dup_empty").to_dicts(), flush=True)
        X, y = band.select(FEATS).to_numpy().astype(np.float32), band["y"].to_numpy()
        fold = (band["s1_id"].hash(seed=5) % 2).to_numpy()
        newp = np.zeros(len(y), dtype=np.float32)
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.05,
                  "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist", "device": "cuda", "seed": 1}
        for k in (0, 1):
            dva = xgb.DMatrix(X[fold == k], y[fold == k], feature_names=FEATS)
            m = xgb.train(params, xgb.DMatrix(X[fold != k], y[fold != k], feature_names=FEATS), 2000,
                          evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
            newp[fold == k] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        pb = np.clip(band["p"].to_numpy(), 1e-6, 1 - 1e-6)

        def ll(p: np.ndarray) -> float:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        print(f"   logloss stack {ll(pb):.4f} -> specialist {ll(newp):.4f}")
        sp = (o.join(band.select("s1_id", "cand_id").with_columns(pl.Series("q", newp)), on=["s1_id", "cand_id"], how="left")
              .with_columns(pl.coalesce("q", "p").alias("p")).drop("q"))
        f0 = evaluate(exclusive_owner_prob(o, "p"), [c])["ef05"][c]
        f1 = evaluate(exclusive_owner_prob(sp, "p"), [c])["ef05"][c]
        print(f"   macro F0.5 (owner prob): stack {f0:.5f} -> + empty-address specialist {f1:.5f} ({f1 - f0:+.5f})", flush=True)


if __name__ == "__main__":
    main()
