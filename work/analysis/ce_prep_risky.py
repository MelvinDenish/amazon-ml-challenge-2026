"""Extra CE routing: CONFIDENT pairs (GBM p >= 0.995) in structurally risky groups.

70-83% of the test selections in the twin groups and ~53% of empty-address selections were never
seen by the cross-encoders (the routing band stopped at 0.995), yet those are exactly the groups
where test has more distractors than train. Route them too, on test AND on the held-out eval S1s
(same rule), so the stacker learns how to use CE scores for confident pairs."""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from labelshift import group_expr  # noqa: E402

A = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts"
OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
GC = ["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty"]


def risky(pred: pl.DataFrame, split: str, countries: tuple) -> pl.DataFrame:
    g = pl.concat([pl.read_parquet(f"{A}/groups/{split}_{c}.parquet", columns=GC) for c in countries]).with_columns(
        group_expr()).filter(pl.col("group") != "rest").select("s1_id", "cand_id", "group")
    return pred.filter(pl.col("p") >= 0.995).join(g, on=["s1_id", "cand_id"])


ev = pl.read_parquet(f"{A}/oof/blend_v7.parquet").filter((pl.col("s1_id").hash(seed=45) % 10_000) < 5000)
e = risky(ev, "train", ("India", "US"))
print("eval risky confident pairs", e.height, "true rate", round(float(e["y"].mean()), 4), e.group_by("group").agg(pl.len(), pl.col("y").mean()).to_dicts())
e.select("s1_id", "cand_id", "country", "y", "p").write_parquet(f"{OUT}/ce_eval_x.parquet")
parts = []
for c in ("France", "India", "US"):
    t = pl.scan_parquet(f"{A}/oof/test_blend_v7.parquet").filter(pl.col("country") == c).collect()
    parts.append(risky(t, "test", (c,)).select("s1_id", "cand_id", "country", "p", "group"))
t = pl.concat(parts)
print("test risky confident pairs", t.height, t.group_by("country", "group").len().sort("country", "group").to_dicts())
t.drop("group").write_parquet(f"{OUT}/ce_test_x.parquet")
