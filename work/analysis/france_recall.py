"""Label-free recall check for France: are 'obvious' matches missing from the candidate set?

Obvious match: an S2/S3 record of the same country whose folded name (accents/case/punctuation
removed, legal forms dropped) equals the S1's folded name AND whose first address number equals the
S1's. Count, per country, the share of such obvious records that are NOT in the candidate table, and
the share that are in it but NOT selected. US/India on train give the reference (true share known),
France on test is the question.
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402

LEGAL = r"\b(sas|sasu|sarl|eurl|sa|sci|snc|ei|cie|inc|llc|ltd|co|corp|corporation|incorporated|lp|llp|pvt|private|limited|company|pc|pllc)\b"
FOLD = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEGAL, " ").str.replace_all(r"\s+", ""))
NUM = pl.col("business_address").fill_null("").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1")


def check(split: str, c: str, pred: pl.DataFrame, n_s1: int = 60000) -> None:
    """Obvious-match counts for a sample of S1s of one split/country."""
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), FOLD.alias("f"), NUM.alias("n"))
    q = q.join(pred.select("s1_id").unique(), on="s1_id", how="semi")
    q = q.sample(min(n_s1, q.height), seed=1)
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), FOLD.alias("f"), NUM.alias("n"))
    ob = (q.filter(pl.col("f").str.len_chars() >= 4).drop_nulls("n").join(t.drop_nulls("n"), on=["f", "n"])
          .select("s1_id", "cand_id").unique())
    inc = ob.join(pred.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi")
    sel = select_expected_f05(one_owner(exclusive_owner_prob(pred, "p"), "p"), "p").select("s1_id", "cand_id")
    chosen = ob.join(sel, on=["s1_id", "cand_id"], how="semi")
    line = (f"[{split} {c}] obvious records/S1 {ob.height / q.height:.3f} | NOT in candidates {1 - inc.height / max(ob.height, 1):.2%} | "
            f"in candidates but not selected {(inc.height - chosen.height) / max(ob.height, 1):.2%}")
    if split == "train":
        gt = load_ground_truth_pairs()
        tr = ob.join(gt, on=["s1_id", "cand_id"], how="semi").height
        line += f" | share truly matching {tr / max(ob.height, 1):.1%}"
    print(line, flush=True)


def main() -> None:
    A = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts"
    test = pl.read_parquet(f"{A}/oof/test_final_v09h.parquet")
    ev = pl.read_parquet(f"{A}/oof/test_final_v09_evaloof.parquet")
    for c in ("US", "India"):
        check("train", c, ev.filter(pl.col("country") == c))
    for c in ("France", "US", "India"):
        check("test", c, test.filter(pl.col("country") == c))


if __name__ == "__main__":
    main()
