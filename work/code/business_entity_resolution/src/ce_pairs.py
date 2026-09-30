"""Build the cross-encoder pair lists (train / held-out eval / test) from GBM predictions.

    python ce_pairs.py --eval-oof blend_v7 --train-oofs xgb_v6,xgb_v7b --test test_blend_v7

S1 universes (all from train, real labels, honest out-of-fold GBM probabilities):
  eval  = S1s of --eval-oof that fall in the deterministic v08 hash sample (features.py rule,
          frac 0.5). Every pair of these S1s is kept for the stacker's full-pipeline evaluation;
          the uncertain ones (0.005 < p < 0.995) are scored by the CE.
  train = every other S1 covered by the --train-oofs, i.e. never an eval S1. CE fine-tuning pairs:
          the uncertain band (0.02 < p < 0.98), all confident mistakes, and 200k easy controls of
          each label.
Routing on test: 0.005 < p < 0.995 under the --test predictions (same rule as eval).
Writes artifacts/ce/{ce_train2,ce_eval,ce_test}.parquet.
"""

import argparse

import polars as pl

from config import ARTIFACT_DIR, SEED

OOF_DIR = ARTIFACT_DIR / "oof"
CE_DIR = ARTIFACT_DIR / "ce"
LO, HI = 0.005, 0.995


def in_v08_sample() -> pl.Expr:
    """features.py sampling rule for frac 0.5 (the v08 training sample)."""
    return (pl.col("s1_id").hash(seed=SEED + 3) % 10_000) < 5000


def main() -> None:
    """Write the three pair lists."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-oof", default="blend_v7")
    ap.add_argument("--train-oofs", default="xgb_v6,xgb_v7b")
    ap.add_argument("--test", default="test_blend_v7")
    a = ap.parse_args()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["s1_id", "cand_id", "country", "y", "p"]

    ev_all = pl.read_parquet(OOF_DIR / f"{a.eval_oof}.parquet").select(cols).filter(in_v08_sample())
    ev_all.filter((pl.col("p") > LO) & (pl.col("p") < HI)).write_parquet(CE_DIR / "ce_eval.parquet")
    eval_s1 = ev_all.select("s1_id").unique()

    parts, seen = [], None
    for name in a.train_oofs.split(","):
        o = pl.read_parquet(OOF_DIR / f"{name}.parquet").select(cols)
        if seen is not None:
            o = o.join(seen, on="s1_id", how="anti")
        parts.append(o)
        seen = o.select("s1_id").unique() if seen is None else pl.concat([seen, o.select("s1_id").unique()])
    pool = pl.concat(parts).join(eval_s1, on="s1_id", how="anti")
    tr = pl.concat([
        pool.filter((pl.col("p") > 0.02) & (pl.col("p") < 0.98)),
        pool.filter((pl.col("p") >= 0.98) & (pl.col("y") == 0)),
        pool.filter((pl.col("p") <= 0.02) & (pl.col("y") == 1)),
        pool.filter((pl.col("p") >= 0.98) & (pl.col("y") == 1)).sample(200_000, seed=1),
        pool.filter((pl.col("p") <= 0.02) & (pl.col("y") == 0)).sample(200_000, seed=2),
    ]).unique(["s1_id", "cand_id"])
    tr.write_parquet(CE_DIR / "ce_train2.parquet")

    t = pl.scan_parquet(OOF_DIR / f"{a.test}.parquet")
    names = t.collect_schema().names()
    pcol = "p" if "p" in names else [c for c in names if c not in ("s1_id", "cand_id", "country")][0]
    (t.filter((pl.col(pcol) > LO) & (pl.col(pcol) < HI)).select("s1_id", "cand_id", "country", pl.col(pcol).alias("p"))
     .collect().write_parquet(CE_DIR / "ce_test.parquet"))
    print(f"eval pairs {ev_all.filter((pl.col('p') > LO) & (pl.col('p') < HI)).height:,} (S1 {eval_s1.height:,}) | "
          f"train pairs {tr.height:,} | written to {CE_DIR}")


if __name__ == "__main__":
    main()
