"""Which ADDED name words mark a different business? Match rate of pairs by token present
only on the candidate side (train OOF, real labels), plus how often each appears on test."""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
c = sys.argv[1]
TOK = pl.col("business_name").str.to_lowercase().str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(
    pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique()


def added(split: str, pred: str, n_s1: int) -> pl.DataFrame:
    p = pl.scan_parquet(f"{K5}/oof/{pred}.parquet").filter(pl.col("country") == c).collect()
    s1 = p.select("s1_id").unique().sort("s1_id").head(n_s1)
    p = p.join(s1, on="s1_id", how="semi").filter(pl.col("p") > 0.02)  # plausible pairs only
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), TOK.alias("qt"))
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"))
    d = p.join(q, on="s1_id").join(t, on="cand_id")
    return d.with_columns(pl.col("tt").list.set_difference(pl.col("qt")).alias("add")).explode("add").drop_nulls("add")


tr = added("train", "blend_v7", 200_000)
te = added("test", "test_blend_v7", 200_000)
stats = tr.group_by("add").agg(pl.len().alias("n_tr"), pl.col("y").mean().round(3).alias("match_rate"),
                               pl.col("p").mean().round(3).alias("mean_p_tr"))
tst = te.group_by("add").agg(pl.len().alias("n_te"), pl.col("p").mean().round(3).alias("mean_p_te"))
j = stats.join(tst, on="add", how="full", coalesce=True).with_columns(pl.col("n_tr", "n_te").fill_null(0))
pl.Config.set_tbl_rows(70)
print(f"[{c}] train added-token rows {tr.height:,} (base match rate {tr['y'].mean():.3f}); test rows {te.height:,}")
print("most frequent added tokens on TEST:")
print(j.sort("n_te", descending=True).head(45))
print("test tokens frequent on test but rare on train (n_te > 5*n_tr):")
print(j.filter((pl.col("n_te") > 200) & (pl.col("n_te") > 5 * pl.col("n_tr"))).sort("n_te", descending=True).head(25))
