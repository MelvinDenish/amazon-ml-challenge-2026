"""Measure the sibling-distractor generator on TEST (vs train) and save a library of deltas.

Sibling-like pair: every S1 name token also appears in the candidate name (after accent/case
folding), the candidate ADDS >= 1 token, and both addresses have a house number that differs.
For each country:
  * sibling-like pairs per S1 on test vs train (train: match rate too),
  * mini-cluster sizes (records of one S1 sharing the same other number + same added words),
  * house-number offset distribution, added-word position (suffix / prefix / inside).
Saves artifacts/sibling_deltas.parquet: one row per test sibling mini-cluster
(country, added words, position, offset, cluster size).
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\sibling_deltas.parquet"
FOLD = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")))
NUM = pl.col("business_address").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1").cast(pl.Int64, strict=False)


def sib_pairs(split: str, pred: str, c: str) -> tuple[pl.DataFrame, int]:
    """Sibling-like pairs among the candidate pairs of one split/country."""
    p = pl.scan_parquet(f"{K5}/oof/{pred}.parquet").filter(pl.col("country") == c).collect()
    if "p" not in p.columns:
        p = p.rename({[x for x in p.columns if x not in ("s1_id", "cand_id", "country", "y")][0]: "p"})
    n = p["s1_id"].n_unique()
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), "business_name", NUM.alias("qn"))
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), "business_name", NUM.alias("tn"))
    # numbers first (cheap), then tokens only for the pairs that can be siblings
    d = (p.select("s1_id", "cand_id", "p", *(["y"] if "y" in p.columns else []))
         .join(q.select("s1_id", "qn"), on="s1_id").join(t.select("cand_id", "tn"), on="cand_id")
         .filter(pl.col("qn").is_not_null() & pl.col("tn").is_not_null() & (pl.col("qn") != pl.col("tn"))))
    del p
    d = (d.join(q.select("s1_id", FOLD.alias("qt")), on="s1_id").join(t.select("cand_id", FOLD.alias("tt")), on="cand_id"))
    d = d.with_columns(rem=pl.col("qt").list.set_difference(pl.col("tt")).list.len(),
                       add=pl.col("tt").list.set_difference(pl.col("qt")))
    d = d.filter((pl.col("rem") == 0) & (pl.col("add").list.len() > 0))
    d = d.with_columns(
        add_txt=pl.col("add").list.sort().list.join(" "),
        pos=pl.when(pl.col("add").list.contains(pl.col("tt").list.last())).then(pl.lit("suffix"))
        .when(pl.col("add").list.contains(pl.col("tt").list.first())).then(pl.lit("prefix")).otherwise(pl.lit("inside")),
        offset=(pl.col("tn") - pl.col("qn")))
    return d.drop("qt", "tt", "rem", "add"), n


def main() -> None:
    libs = []
    for c in ("US", "India", "France"):
        te, n_te = sib_pairs("test", "test_blend_v7", c)
        clus = te.group_by("s1_id", "tn", "add_txt").agg(pl.len().alias("size"), pl.first("pos"), pl.first("offset"))
        line = f"[{c}] TEST sibling-like pairs/S1 {te.height / n_te:.3f}  mini-clusters/S1 {clus.height / n_te:.3f}"
        if c != "France":
            tr, n_tr = sib_pairs("train", "blend_v7", c)
            ctr = tr.group_by("s1_id", "tn", "add_txt").agg(pl.len().alias("size"), pl.col("y").max().alias("ymax"))
            line += (f" | TRAIN pairs/S1 {tr.height / n_tr:.3f} (match rate {tr['y'].mean():.3f}) "
                     f"mini-clusters/S1 {ctr.height / n_tr:.3f} (any-true {ctr['ymax'].mean():.3f})")
        print(line, flush=True)
        print("  cluster size:", clus["size"].value_counts().sort("size").head(6).to_dicts())
        print("  position:", clus["pos"].value_counts().sort("count", descending=True).to_dicts())
        off = clus["offset"]
        print("  |offset| quantiles 10/25/50/75/90:", [int(off.abs().quantile(x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)],
              " share positive", round(float((off > 0).mean()), 3))
        print("  top added words:", clus["add_txt"].value_counts().sort("count", descending=True).head(25).to_dicts(), flush=True)
        libs.append(clus.select(pl.lit(c).alias("country"), "add_txt", "pos", "offset", "size"))
    pl.concat(libs).write_parquet(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
