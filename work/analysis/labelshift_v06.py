"""Label-shift (EM prior) check on the v06 OOF (item 2 of the review follow-up).

1. Synthetic shift: labelshift.validate (drop true pairs inside a group; EM must recover the prior).
2. Natural shift: per group, pretend India is 'train' and US is 'target' (and reverse).
   EM gets only the target's probabilities plus the source prior. This tests EM where the
   feature distributions ALSO differ, which is the real test-time situation.
3. Does rescaling help? Apply the EM-adjusted probabilities to the target country and
   re-score the full pipeline against the unadjusted one.
"""
import os
import sys

os.environ["BER_ARTIFACT_DIR"] = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k2_out_v2\artifacts"
sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import config  # noqa: E402
config.PARQUET_DIR = config.Path(r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\data")
import io_utils  # noqa: E402
io_utils.PARQUET_DIR = config.PARQUET_DIR
from labelshift import em_prior, load, validate  # noqa: E402
from train import evaluate  # noqa: E402

GROUPS = ("form conflict", "near-number twin", "other-number twin", "empty cand address")


def main() -> None:
    tr = load("train", "xgb_v6")
    print("1) SYNTHETIC SHIFT")
    validate(tr)
    print("\n2) NATURAL SHIFT (source -> target country)")
    for src, tgt in (("India", "US"), ("US", "India")):
        adj = []
        for g in GROUPS + ("rest",):
            s = tr.filter((pl.col("country") == src) & (pl.col("group") == g))
            t = tr.filter((pl.col("country") == tgt) & (pl.col("group") == g))
            pi_s, pi_t = float(s["y"].mean()), float(t["y"].mean())
            est, q = em_prior(t["p"].to_numpy(), pi_s)
            print(f"  [{src}->{tgt} {g:18s}] n={t.height:>9,} source prior {pi_s:.4f}  TRUE target {pi_t:.4f}  "
                  f"EM {est:.4f}  mean p {t['p'].mean():.4f}")
            if g != "rest":
                adj.append(t.select("s1_id", "cand_id", pl.Series("p_em", q.astype(np.float32))))
        t_all = tr.filter(pl.col("country") == tgt).select("s1_id", "cand_id", "country", "y", "p")
        t_adj = t_all.join(pl.concat(adj), on=["s1_id", "cand_id"], how="left").with_columns(
            pl.coalesce("p_em", "p").alias("p")).drop("p_em")
        print(f"  {tgt} full pipeline: unadjusted {evaluate(t_all, [tgt])['ef05'][tgt]:.6f}  "
              f"EM-adjusted groups {evaluate(t_adj, [tgt])['ef05'][tgt]:.6f}", flush=True)


if __name__ == "__main__":
    main()
