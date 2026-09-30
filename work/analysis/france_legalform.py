"""France-specific legal-form identity rule (test only; France has no labels).

In the France test data the dominant distractor is the legal-form sibling ('X SARL' vs 'X SAS/SCI/
SNC/EURL' at nearby numbers or with no address) and the models, trained on US/India where legal forms
are noise, cannot separate same-name S1s by legal form: France has 39 records per 1k S1 claimed
(p>=0.3) by >=2 S1s vs 2 per 1k on train, and 115 per 1k S1 with p>=0.5 left unselected (US 21).
Rule for France pairs: if both names carry a legal form and the sets are disjoint, multiply the odds
by FACTOR before exclusive-owner normalisation. A dropped/added form is left neutral (true copies
often drop the form). Contested records then go to the claimant whose form matches.

    python france_legalform.py <test_pred.parquet> <out.parquet> [FACTOR]
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402

FORMS = r"\b(sas|sasu|sarl|eurl|sa|sci|snc|ei|cie)\b"
LF = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
      .str.replace_all(r"[^a-z ]", " ").str.replace_all(r"\bs a r l\b", "sarl").str.replace_all(r"\bs a s u\b", "sasu")
      .str.replace_all(r"\bs a s\b", "sas").str.replace_all(r"\be i\b", "ei").str.replace_all(r"\bs c i\b", "sci")
      .str.replace_all(r"\bs n c\b", "snc").str.replace_all(r"\bs a\b", "sa").str.extract_all(FORMS).list.unique())


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    factor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    t = pl.read_parquet(src)
    fr = t.filter(pl.col("country") == "France")
    rec = pl.concat([load_records("test", s, "France") for s in ("source1", "source2", "source3")]).select("entity_id", LF.alias("lf"))
    d = fr.join(rec.rename({"entity_id": "s1_id", "lf": "lq"}), on="s1_id", how="left").join(rec.rename({"entity_id": "cand_id", "lf": "lt"}), on="cand_id", how="left")
    conflict = ((pl.col("lq").list.len() > 0) & (pl.col("lt").list.len() > 0)
                & (pl.col("lq").list.set_intersection(pl.col("lt")).list.len() == 0))
    o = pl.col("p").cast(pl.Float64).clip(1e-7, 1 - 1e-7)
    o = o / (1 - o) * factor
    d = d.with_columns(conflict=conflict.fill_null(False)).with_columns(
        pl.when(pl.col("conflict")).then(o / (1 + o)).otherwise(pl.col("p")).cast(pl.Float32).alias("p2"))
    n = fr["s1_id"].n_unique()
    b = select_expected_f05(one_owner(exclusive_owner_prob(fr, "p"), "p"), "p").select("s1_id", "cand_id")
    a = select_expected_f05(one_owner(exclusive_owner_prob(d.select("s1_id", "cand_id", "country", pl.col("p2").alias("p")), "p"), "p"), "p").select("s1_id", "cand_id")
    rem = b.join(a, on=["s1_id", "cand_id"], how="anti").height
    add = a.join(b, on=["s1_id", "cand_id"], how="anti").height
    print(f"France legal-form rule (factor {factor}): pairs with conflicting forms {d['conflict'].mean():.1%} of France pairs; "
          f"selections removed {rem / n * 1000:.1f}/1k S1, added {add / n * 1000:.1f}/1k S1; matches/S1 {b.height / n:.3f} -> {a.height / n:.3f}")
    new = pl.concat([t.filter(pl.col("country") != "France"), d.select("s1_id", "cand_id", "country", pl.col("p2").alias("p"))])
    new.write_parquet(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
