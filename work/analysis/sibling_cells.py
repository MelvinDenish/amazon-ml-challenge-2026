"""Language-agnostic sibling-distractor structure: match rate by (added-word type x number relation x anchor).

Test (all 3 countries) has many '<S1 name> + extra word' records at a nearby house number next to a
confident cluster at the S1's own number. The words differ by language (Holdings / Développement),
the structure does not. Cells, per plausible pair (p > 0.02):
  added  : none | legal form only | other word          (accent-stripped name tokens, t minus q)
  number : same | differs | missing                      (first number of the raw address, zeros stripped)
  anchor : S1 has ANOTHER candidate with p >= 0.9 whose number equals the S1's number
Train gives the real match rate per cell; test gives counts, mean p and share currently selected (p>0.5).
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
LEGAL = ["inc", "llc", "ltd", "co", "corp", "corporation", "incorporated", "lp", "llp", "pvt", "private", "limited",
         "company", "plc", "pllc", "pc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "cie", "l", "c", "p"]
TOK = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
       .str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique())
NUM = (pl.col("business_address").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1"))


def cells(split: str, pred: str, c: str, n_s1: int | None) -> pl.DataFrame:
    p = pl.scan_parquet(f"{K5}/oof/{pred}.parquet").filter(pl.col("country") == c).collect()
    if "p" not in p.columns:
        p = p.rename({[x for x in p.columns if x not in ("s1_id", "cand_id", "country", "y")][0]: "p"})
    if n_s1:
        p = p.join(p.select("s1_id").unique().sort("s1_id").head(n_s1), on="s1_id", how="semi")
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), TOK.alias("qt"), NUM.alias("qn"))
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"), NUM.alias("tn"))
    d = p.join(q, on="s1_id").join(t, on="cand_id", how="left")
    d = d.with_columns(
        same=(pl.col("qn").is_not_null() & (pl.col("qn") == pl.col("tn"))).fill_null(False),
        add=pl.col("tt").list.set_difference(pl.col("qt")))
    d = d.with_columns(anchor_n=(pl.col("same") & (pl.col("p") >= 0.9)).cast(pl.Int32).sum().over("s1_id"))
    d = d.filter(pl.col("p") > 0.02).with_columns(
        added=pl.when(pl.col("add").list.len() == 0).then(pl.lit("none"))
        .when(pl.col("add").list.eval(pl.element().is_in(LEGAL)).list.all()).then(pl.lit("legal only"))
        .otherwise(pl.lit("other word")),
        number=pl.when(pl.col("qn").is_null() | pl.col("tn").is_null()).then(pl.lit("missing"))
        .when(pl.col("same")).then(pl.lit("same")).otherwise(pl.lit("differs")),
        anchor=((pl.col("anchor_n") - (pl.col("same") & (pl.col("p") >= 0.9)).cast(pl.Int32)) >= 1))
    n = p["s1_id"].n_unique()
    agg = [pl.len().alias("n"), (pl.len() / n * 1000).round(1).alias("per_1k_S1"), pl.col("p").mean().round(3).alias("mean_p"),
           (pl.col("p") > 0.5).mean().round(3).alias("sel_share")]
    if "y" in d.columns:
        agg.append(pl.col("y").mean().round(3).alias("match_rate"))
    return d.group_by("added", "number", "anchor").agg(agg).sort("added", "number", "anchor")


pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
for c in ("US", "India"):
    print(f"\n######## TRAIN {c}")
    print(cells("train", "blend_v7", c, 150_000))
for c in ("US", "India", "France"):
    print(f"\n######## TEST {c}")
    print(cells("test", "test_blend_v7", c, 150_000))
