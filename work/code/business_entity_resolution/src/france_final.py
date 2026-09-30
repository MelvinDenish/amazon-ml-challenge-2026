"""Assemble the final test predictions: US/India from the 8-encoder stack, France from the France recipe.

    python france_final.py --main test_v10_ce8 --fr-base test_fr3 --fr-ml test_fr5 --out test_final

France (no labels; every step below was scored on the public leaderboard with US/India rows fixed):
  1. p from the 3-encoder stack with the France-adapted XLM-R (v1, v2, xlmr-fr; ce_stack --no-src-feats)
     plus a +1 logit push (LB 0.983929).
  2. Generator filler variants ('X SARL Développement' for 'X Santé SARL', same number and street, no S1
     carrying the record's exact name): p = max(p, p of the multilingual France-adapted 5-encoder stack).
  3. Co-located sibling cap (swap_rules.vocab_swap_cap, extended): LB 0.985752 with the original rule,
     0.986229 with the extension and step 2.
"""
import argparse

import numpy as np
import polars as pl

from config import ARTIFACT_DIR
from swap_rules import filler_swap_pairs, vocab_swap_cap

OOF_DIR = ARTIFACT_DIR / "oof"


def logit_push(p: pl.Series, shift: float) -> pl.Series:
    """sigmoid(logit(p) + shift), computed in float64."""
    x = np.clip(p.to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
    return pl.Series(1.0 / (1.0 + np.exp(-(np.log(x / (1 - x)) + shift))), dtype=pl.Float32)


def france(base: pl.DataFrame, ml: pl.DataFrame, push: float) -> pl.DataFrame:
    """France rows of the final prediction (s1_id, cand_id, country, p)."""
    d = base.with_columns(p=logit_push(base["p"], push))
    fill = filler_swap_pairs(d, "test", "France").with_columns(_f=pl.lit(True))
    d = (d.join(ml.select("s1_id", "cand_id", pl.col("p").cast(pl.Float32).alias("_pm")), on=["s1_id", "cand_id"], how="left")
         .join(fill, on=["s1_id", "cand_id"], how="left")
         .with_columns(pl.when(pl.col("_f")).then(pl.max_horizontal("p", "_pm")).otherwise(pl.col("p")).cast(pl.Float32).alias("p"))
         .drop("_pm", "_f"))
    return vocab_swap_cap(d, "test", "France", extended=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True, help="stacked test predictions used for US/India")
    ap.add_argument("--fr-base", required=True, help="3-encoder France-adapted stack")
    ap.add_argument("--fr-ml", required=True, help="5-encoder multilingual France-adapted stack")
    ap.add_argument("--push", type=float, default=1.0)
    ap.add_argument("--gbm", default="test_blend_v7", help="stage-1 GBM test predictions (cascade filter)")
    ap.add_argument("--prune", type=float, default=0.005,
                    help="keep only blocked pairs with GBM p > prune: the cascade's candidate set (0 = keep all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = ["s1_id", "cand_id", "country", "p"]
    # Cascade: blocking keeps ~26 pairs per S1; the GBM discards those at p <= 0.005 (the cross-encoder
    # routing threshold, so these pairs never reach the encoders and can never be selected). The ~5 pairs
    # per S1 left are the candidate set scored by the encoders, the stacker and the final assignment.
    keep = (pl.scan_parquet(OOF_DIR / f"{a.gbm}.parquet").filter(pl.col("p") > a.prune).select("s1_id", "cand_id")
            if a.prune > 0 else None)

    def load(name: str, france_only: bool) -> pl.LazyFrame:
        lf = pl.scan_parquet(OOF_DIR / f"{name}.parquet").filter(
            (pl.col("country") == "France") if france_only else (pl.col("country") != "France"))
        lf = lf.select(cols).with_columns(pl.col("p").cast(pl.Float32))
        return lf.join(keep, on=["s1_id", "cand_id"], how="semi") if keep is not None else lf

    f = france(load(a.fr_base, True).collect(), load(a.fr_ml, True).collect(), a.push).select(cols)
    pl.concat([load(a.main, False), f.lazy()]).sink_parquet(OOF_DIR / f"{a.out}.parquet")
    print("wrote", OOF_DIR / f"{a.out}.parquet")


if __name__ == "__main__":
    main()
