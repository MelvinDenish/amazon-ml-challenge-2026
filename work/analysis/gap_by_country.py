"""Where is the train->test gap? Label-free expected F0.5 per country (v07 blend + owner prob).

For each S1 the pipeline's selected set S and the probabilities of all its (owned)
candidates give the model's own expected F0.5 (same ratio-of-expectations formula the
selector maximises). On train we also have the real F0.5, so the self-estimate's bias is
measured there and the test estimates are read against it.
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from score import per_entity_f05  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"


def expected(pred: pl.DataFrame) -> pl.DataFrame:
    """Per S1: expected F0.5 of the chosen set, expected |truth| and uncertainty mass."""
    owned = one_owner(exclusive_owner_prob(pred, "p"), "p")
    sel = select_expected_f05(owned, "p").select("s1_id", "cand_id").with_columns(pl.lit(1).alias("sel"))
    d = owned.join(sel, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("sel").fill_null(0))
    g = d.group_by("s1_id").agg(
        pl.col("p").sum().alias("et"), (pl.col("p") * pl.col("sel")).sum().alias("tp"), pl.col("sel").sum().alias("m"),
        (1 - pl.col("p")).clip(1e-6, 1).log().sum().exp().alias("p_none"),
        ((pl.col("p") > 0.1) & (pl.col("p") < 0.9)).sum().alias("n_unsure"))
    return g.with_columns(
        pl.when(pl.col("m") == 0).then(pl.col("p_none"))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("et") / 0.985 + pl.col("m"))).alias("ef"))


def main() -> None:
    oof = pl.read_parquet(f"{K5}/oof/blend_v7.parquet")
    gt = load_ground_truth_pairs()
    for c in ("India", "US"):
        o = oof.filter(pl.col("country") == c)
        e = expected(o.select("s1_id", "cand_id", "p"))
        owned = one_owner(exclusive_owner_prob(o, "p"), "p")
        sel = select_expected_f05(owned, "p")
        truth = gt.join(o.select("s1_id").unique(), on="s1_id", how="semi")
        real = per_entity_f05(truth, sel, o["s1_id"].unique())
        print(f"TRAIN {c:6s} S1 {e.height:,}  self-estimate {e['ef'].mean():.4f}  REAL {real['f05'].mean():.4f}  "
              f"unsure pairs/S1 {e['n_unsure'].mean():.3f}", flush=True)
    for c in ("France", "India", "US"):
        t = (pl.scan_parquet(f"{K5}/oof/test_blend_v7.parquet").filter(pl.col("country") == c)
             .select("s1_id", "cand_id", "p").collect())
        e = expected(t)
        print(f"TEST  {c:6s} S1 {e.height:,}  self-estimate {e['ef'].mean():.4f}  unsure pairs/S1 {e['n_unsure'].mean():.3f}  "
              f"low-ef S1s (<0.8) {(e['ef'] < 0.8).mean():.3%}", flush=True)


if __name__ == "__main__":
    main()
