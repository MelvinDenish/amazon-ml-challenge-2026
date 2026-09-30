"""Show test US S1s with unsure near-number twins: all their candidates with p, raw text."""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402
from labelshift import group_expr  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
split, c = sys.argv[1], sys.argv[2]
pred = "blend_v7" if split == "train" else "test_blend_v7"
p = pl.scan_parquet(f"{K5}/oof/{pred}.parquet").filter(pl.col("country") == c).collect()
g = pl.read_parquet(f"{K5}/groups/{split}_{c}.parquet").with_columns(group_expr())
d = p.join(g, on=["s1_id", "cand_id"])
hit = d.filter((pl.col("group") == "near-number twin") & (pl.col("p") > 0.1) & (pl.col("p") < 0.9))
s1s = hit["s1_id"].unique().sort().sample(8, seed=3)
rec = pl.concat([load_records(split, s, c).select("entity_id", "business_name", "business_address") for s in ("source1", "source2", "source3")])
for s in s1s.to_list():
    r = rec.filter(pl.col("entity_id") == s).row(0)
    print(f"\n=== {r[1]} | {r[2]}")
    x = d.filter(pl.col("s1_id") == s).sort("p", descending=True).head(8).join(rec, left_on="cand_id", right_on="entity_id")
    for y in x.sort("p", descending=True).iter_rows(named=True):
        lab = f" y={y['y']}" if "y" in y else ""
        print(f"   p={y['p']:.3f}{lab} {y['group'][:12]:12s} {y['business_name'][:45]:45s} | {y['business_address'][:60]}")
