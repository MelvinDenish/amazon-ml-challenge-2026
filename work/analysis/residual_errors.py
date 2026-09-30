"""Where do errors remain after the CE stack? Full pipeline on the CE-stacked OOF (eval S1s):
FP / missed-in-candidates by group and by probability band, plus the candidate ceiling."""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs  # noqa: E402
from labelshift import group_expr  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from score import per_entity_f05  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
oof = pl.read_parquet(sys.argv[1])
gt = load_ground_truth_pairs()
for c in ("India", "US"):
    o = oof.filter(pl.col("country") == c)
    s1 = o["s1_id"].unique()
    truth = gt.join(o.select("s1_id").unique(), on="s1_id", how="semi")
    sel = select_expected_f05(one_owner(exclusive_owner_prob(o, "p"), "p"), "p").select("s1_id", "cand_id")
    base = per_entity_f05(truth, sel, s1)["f05"].mean()
    oracle = per_entity_f05(truth, o.filter(pl.col("y") == 1).select("s1_id", "cand_id"), s1)["f05"].mean()
    no_fp = per_entity_f05(truth, sel.join(truth, on=["s1_id", "cand_id"], how="semi"), s1)["f05"].mean()
    in_cand = o.filter(pl.col("y") == 1).select("s1_id", "cand_id")
    no_fn = per_entity_f05(truth, pl.concat([sel.join(truth, on=["s1_id", "cand_id"], how="semi"), in_cand]).unique(), s1)["f05"].mean()
    print(f"[{c}] S1 {len(s1):,}  F0.5 {base:.5f} | remove all FP -> {no_fp:.5f} (+{no_fp - base:.5f}) | "
          f"perfect on candidates {oracle:.5f} | FP-free + all in-cand FN recovered {no_fn:.5f}")
    g = pl.read_parquet(f"{K5}/groups/train_{c}.parquet", columns=["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty"]).with_columns(group_expr()).select("s1_id", "cand_id", "group")
    d = o.join(sel.with_columns(pl.lit(1).alias("sel")), on=["s1_id", "cand_id"], how="left").with_columns(pl.col("sel").fill_null(0)).join(g, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("group").fill_null("rest"))
    d = d.with_columns(band=pl.when(pl.col("p") < 0.005).then(pl.lit("p<0.005")).when(pl.col("p") < 0.5).then(pl.lit("0.005-0.5"))
                       .when(pl.col("p") < 0.995).then(pl.lit("0.5-0.995")).otherwise(pl.lit(">=0.995")))
    n = len(s1)
    fp = d.filter((pl.col("sel") == 1) & (pl.col("y") == 0)).group_by("group", "band").len().with_columns((pl.col("len") / n * 1000).round(2).alias("FP_per_1k"))
    fn = d.filter((pl.col("sel") == 0) & (pl.col("y") == 1)).group_by("group", "band").len().with_columns((pl.col("len") / n * 1000).round(2).alias("FN_per_1k"))
    pl.Config.set_tbl_rows(40)
    print(fp.join(fn, on=["group", "band"], how="full", coalesce=True).select("group", "band", "FP_per_1k", "FN_per_1k").sort("group", "band"))
