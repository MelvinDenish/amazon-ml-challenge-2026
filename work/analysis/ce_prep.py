"""Prepare pair lists for the cross-encoder job (review item C).

S1 universe = the v07 blend OOF S1s (v06/v07b overlap, real labels, honest OOF p).
  CE-train S1s: NOT in the v08 deterministic sample  -> CE fine-tuning pairs
  CE-eval  S1s: IN the v08 sample                  -> CE scores for stacking + full-pipeline
                                                        evaluation against v07 AND v08 OOF
Routing (which pairs the CE scores): uncertain pairs (lo < p < 1-lo) plus every pair of
structurally risky groups is too many; the band is widened instead so a later model's
band (v08) is still covered.
"""
import polars as pl

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
SEED = 42
v08 = (pl.col("s1_id").hash(seed=SEED + 3) % 10_000) < 5000  # features.py sampling rule, frac 0.5

oof = pl.read_parquet(f"{K5}/oof/blend_v7.parquet").with_columns(v08.alias("in_v08"))
print("OOF S1s", oof["s1_id"].n_unique(), "in v08 sample share", round(float(oof.unique("s1_id")["in_v08"].mean()), 3))
tr = oof.filter(~pl.col("in_v08"))
band = (pl.col("p") > 0.02) & (pl.col("p") < 0.98)
train = pl.concat([
    tr.filter(band),
    tr.filter((pl.col("p") >= 0.98) & (pl.col("y") == 0)),                 # confident false merges
    tr.filter((pl.col("p") <= 0.02) & (pl.col("y") == 1)),                 # confident misses
    tr.filter((pl.col("p") >= 0.98) & (pl.col("y") == 1)).sample(60_000, seed=1),
    tr.filter((pl.col("p") <= 0.02) & (pl.col("y") == 0)).sample(60_000, seed=2),
]).unique(["s1_id", "cand_id"]).select("s1_id", "cand_id", "country", "y", "p")
print("CE train pairs", train.height, "pos rate", round(float(train["y"].mean()), 3), train.group_by("country").len().to_dicts())
ev = oof.filter(pl.col("in_v08") & (pl.col("p") > 0.005) & (pl.col("p") < 0.995)).select("s1_id", "cand_id", "country", "y", "p")
print("CE eval pairs", ev.height, "S1s", ev["s1_id"].n_unique())
train.write_parquet(f"{OUT}/ce_train.parquet")
ev.write_parquet(f"{OUT}/ce_eval.parquet")
parts = []
for c in ("France", "India", "US"):
    t = (pl.scan_parquet(f"{K5}/oof/test_blend_v7.parquet").filter(pl.col("country") == c)
         .filter((pl.col("p") > 0.005) & (pl.col("p") < 0.995)).select("s1_id", "cand_id", "country", "p").collect())
    parts.append(t)
    print("test", c, t.height)
pl.concat(parts).write_parquet(f"{OUT}/ce_test.parquet")
