"""Which kinds of pairs carry the extra test uncertainty? Unsure (0.1<p<0.9) pairs per 1k S1 by group."""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from labelshift import group_expr  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
COLS = ["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty", "n_tset", "a_tset"]
rows = []
for split, pred in (("train", "blend_v7"), ("test", "test_blend_v7")):
    for c in (("India", "US") if split == "train" else ("France", "India", "US")):
        p = pl.scan_parquet(f"{K5}/oof/{pred}.parquet").filter(pl.col("country") == c).select("s1_id", "cand_id", "p").collect()
        g = pl.read_parquet(f"{K5}/groups/{split}_{c}.parquet", columns=COLS).with_columns(group_expr())
        d = p.join(g, on=["s1_id", "cand_id"])
        n = p["s1_id"].n_unique()
        u = d.filter((pl.col("p") > 0.1) & (pl.col("p") < 0.9))
        u = u.with_columns(pl.when(pl.col("group") != "rest").then(pl.col("group"))
                           .when(pl.col("n_tset") < 0.6).then(pl.lit("rest: name differs"))
                           .when(pl.col("a_tset") < 0.6).then(pl.lit("rest: addr differs"))
                           .otherwise(pl.lit("rest: both similar")).alias("g2"))
        for r in u.group_by("g2").len().to_dicts():
            rows.append({"split_country": f"{split}_{c}", "group": r["g2"], "per_1k": round(r["len"] / n * 1000, 1)})
        del p, g, d
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).pivot(on="split_country", index="group", values="per_1k").sort("group"))
