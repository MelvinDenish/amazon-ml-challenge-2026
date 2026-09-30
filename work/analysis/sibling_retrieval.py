"""Transitive recall: do missed true records look like the S1's OTHER (retrieved, true) records?

For S1s of the v07 OOF sample (real labels): siblings = true records already in the candidate set.
New candidates = S2/S3 records NOT in the candidate set that share an exact normalised name or an
exact normalised (non-trivial) address with a sibling. Reports how many true misses this recovers
and how many false records it drags in (precision of the route), per country.
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
NK = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
      .str.replace_all(r"[^\p{L}\p{N}]", ""))
AK = (pl.col("business_address").fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
      .str.replace_all(r"[^\p{L}\p{N}]", ""))


def main() -> None:
    gt = load_ground_truth_pairs()
    for c in ("India", "US"):
        oof = pl.scan_parquet(f"{K5}/oof/blend_v7.parquet").filter(pl.col("country") == c).select("s1_id", "cand_id", "y").collect()
        s1 = oof.select("s1_id").unique()
        truth = gt.join(s1, on="s1_id")
        cand = oof.select("s1_id", "cand_id")
        miss = truth.join(cand, on=["s1_id", "cand_id"], how="anti")
        sib = oof.filter(pl.col("y") == 1).select("s1_id", pl.col("cand_id").alias("sib"))
        t = pl.concat([load_records("train", s, c) for s in ("source2", "source3")]).select(
            pl.col("entity_id").alias("cand_id"), NK.alias("nk"), AK.alias("ak"))
        sk = sib.join(t.rename({"cand_id": "sib"}), on="sib")
        by_n = sk.filter(pl.col("nk").str.len_chars() >= 4).select("s1_id", "nk").unique().join(
            t.select("cand_id", "nk"), on="nk").select("s1_id", "cand_id")
        by_a = sk.filter(pl.col("ak").str.len_chars() >= 12).select("s1_id", "ak").unique().join(
            t.select("cand_id", "ak"), on="ak").select("s1_id", "cand_id")
        n1 = s1.height
        for tag, r in (("same name as a true sibling", by_n), ("same address as a true sibling", by_a),
                       ("either", pl.concat([by_n, by_a]))):
            r = r.unique().join(cand, on=["s1_id", "cand_id"], how="anti")
            hit = r.join(miss, on=["s1_id", "cand_id"], how="semi").height
            print(f"[{c}] {tag:32s}: new pairs {r.height / n1:.3f}/S1, true {hit:,} ({hit / max(r.height, 1):.1%} precision), "
                  f"recovers {hit / miss.height:.1%} of {miss.height:,} misses", flush=True)


if __name__ == "__main__":
    main()
