"""Evidence-driven extra pair features (joined onto the base feature shards).

    python extra_features.py --split train      # for the pairs in feats/train_*.parquet
    python extra_features.py --split test

Each group targets an error class measured on train labels:

  Legal form  (train: LLC->Inc/Co/Ltd/Corp and India Ltd->Pvt Ltd are 0% matches;
               dropping a form, e.g. Pvt Ltd->Pvt, is noise with 96% matches).
               The base normaliser strips legal words, so the model never saw them.
               Forms are mapped to canonical classes (country-agnostic, incl. the
               French SAS/SARL/SASU/EURL/SNC/SA/EI/SCI) and compared as sets:
               added forms, dropped forms, disjoint sets.
  House number (train: same-name candidates at a *different* number are 4-13%
               matches; truncations like 650->65 are ~57%).
               Numeric difference, prefix/truncation, digit edit distance.
  Twin competition: whether the S1's own number is carried by other same-name
               candidates, how many same-name candidates share this candidate's
               number, and whether another S1 has this record at its exact number.
"""

import argparse
import time

import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from build_candidates import CAND_DIR, countries_of
from config import ARTIFACT_DIR
from dataset import load_country
from features import FEAT_DIR, ID_COLS

EXTRA_DIR = ARTIFACT_DIR / "feats_extra"

FORM_CLASSES = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "ltd": "ltd", "limited": "ltd", "llp": "llp", "plc": "plc", "pc": "pc", "pllc": "pllc",
    "lp": "lp", "co": "co", "company": "co", "pvt": "pvt", "private": "pvt", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "snc": "snc", "sa": "sa",
    "ei": "ei", "sci": "sci",
}
HI_NAME = 0.9


def _forms(col: str) -> pl.Expr:
    """Canonical legal-form classes present in a cleaned (space-separated) name."""
    keys, vals = list(FORM_CLASSES), list(FORM_CLASSES.values())
    return (
        pl.col(col).str.split(" ")
        .list.eval(pl.element().filter(pl.element().is_in(keys)).replace(keys, vals))
        .list.unique()
    )


# v2 (from the v06 error review): the normaliser keeps only one house-number token
# and one street word, so "Flat No A-5/101" vs "A-5/114", "30-25-4/1" vs "30-25-4/4",
# "1A" vs "12A Amethyst Ct" looked identical. These read the RAW address instead.
_UNIT_RE = (r"(?:flat|unit|apt|apartment|suite|ste|room|rm|office|shop|door|plot|floor|bldg|building|block)"
            r"\.?\s*(?:no\.?|number|#)?\s*[:#]?\s*([a-z]?-?\d+[a-z]?(?:[-/][a-z0-9]+)*)")
_NUM_TOK = r"\b[a-z]?-?\d+[a-z]*(?:[-/][a-z0-9]+)*"


def _raw_addr_attrs(prefix: str) -> list[pl.Expr]:
    """Compound number tokens and unit number from the raw address."""
    # No postcode handling: measured on the data, US addresses carry no ZIP (5-digit
    # tokens are house numbers, ~10% of records), France has a postcode in <1% and
    # India a PIN in ~0%, so every digit token is kept. Zero padding is stripped
    # ("0520" == "520").
    raw = pl.col("business_address").str.to_lowercase().str.replace_all(r"\b0+(\d)", "$1")
    toks = raw.str.extract_all(_NUM_TOK).list.eval(pl.element().str.strip_chars_start("-")).list.unique()
    return [toks.alias(f"{prefix}_numtoks"), raw.str.extract(_UNIT_RE).str.strip_chars_start("-").alias(f"{prefix}_unit")]


def record_attrs(q: pl.DataFrame, t: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-record house number, legal forms, raw-address number parts and name tokens."""
    qa = q.select(pl.col("entity_id").alias("s1_id"), pl.col("a_num").alias("q_num"), _forms("n_full").alias("q_forms"),
                  *_raw_addr_attrs("q"), pl.col("n_toks").list.unique().alias("q_ntoks"))
    ta = t.select(pl.col("entity_id").alias("cand_id"), pl.col("a_num").alias("t_num"), _forms("n_full").alias("t_forms"),
                  *_raw_addr_attrs("t"), pl.col("n_toks").list.unique().alias("t_ntoks"))
    return qa, ta


def cross_s1_exact(split: str, country: str, qa: pl.DataFrame, ta: pl.DataFrame) -> pl.DataFrame:
    """For each candidate record: how many S1s (over the FULL candidate table) have a
    same-name pair with it at exactly their own house number."""
    c = pl.read_parquet(CAND_DIR / f"{split}_{country}.parquet", columns=["s1_id", "cand_id", "sim_n"])
    c = c.join(qa.select("s1_id", "q_num"), on="s1_id").join(ta.select("cand_id", "t_num"), on="cand_id")
    exact = (pl.col("t_num") != "") & (pl.col("t_num") == pl.col("q_num")) & (pl.col("sim_n") >= HI_NAME)
    return c.with_columns(self_exact=exact.cast(pl.Int32)).with_columns(
        cand_exact_s1=pl.col("self_exact").sum().over("cand_id")
    ).select(ID_COLS + ["self_exact", "cand_exact_s1"])


def build_extra(split: str, country: str) -> pl.DataFrame:
    """Compute the extra features for exactly the pairs in the base feature shard."""
    q, t = load_country(split, country)
    qa, ta = record_attrs(q, t)
    del q, t
    base = pl.read_parquet(FEAT_DIR / f"{split}_{country}.parquet", columns=ID_COLS + ["n_tset", "a_tset"])
    x = base.join(qa, on="s1_id", how="left").join(ta, on="cand_id", how="left")
    x = x.join(cross_s1_exact(split, country, qa, ta), on=ID_COLS, how="left")

    qn, tn = pl.col("q_num"), pl.col("t_num")
    qi = qn.str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
    ti = tn.str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
    both = (qn != "") & (tn != "")
    lev = process.cpdist(x["q_num"].fill_null("").to_list(), x["t_num"].fill_null("").to_list(),
                         scorer=Levenshtein.distance, workers=-1)
    x = x.with_columns(pl.Series("num_digit_lev", lev).cast(pl.Float32))

    hi = pl.col("n_tset") >= HI_NAME
    x = x.with_columns(
        # house-number relation
        num_absdiff=pl.when(both).then((qi - ti).abs()).otherwise(None).cast(pl.Float32),
        num_reldiff=pl.when(both & (qi > 0)).then((qi - ti).abs() / qi).otherwise(None).cast(pl.Float32),
        num_prefix=(both & (qn != tn) & (qn.str.starts_with(tn) | tn.str.starts_with(qn))).cast(pl.UInt8),
        num_len_diff=pl.when(both).then(qn.str.len_chars().cast(pl.Int32) - tn.str.len_chars().cast(pl.Int32))
        .otherwise(None).cast(pl.Float32),
        num_digit_lev=pl.when(both).then(pl.col("num_digit_lev")).otherwise(None),
        # legal forms
        form_q_n=pl.col("q_forms").list.len().cast(pl.UInt8),
        form_t_n=pl.col("t_forms").list.len().cast(pl.UInt8),
        form_added=pl.col("t_forms").list.set_difference(pl.col("q_forms")).list.len().cast(pl.UInt8),
        form_dropped=pl.col("q_forms").list.set_difference(pl.col("t_forms")).list.len().cast(pl.UInt8),
        form_disjoint=((pl.col("q_forms").list.len() > 0) & (pl.col("t_forms").list.len() > 0)
                       & (pl.col("q_forms").list.set_intersection(pl.col("t_forms")).list.len() == 0)).cast(pl.UInt8),
        # twin competition inside this S1's candidate set (same-name candidates only)
        _hi_exact=(hi & both & (qn == tn)).cast(pl.Int32),
        _hi=hi.cast(pl.Int32),
    ).with_columns(
        s1_num_support=pl.col("_hi_exact").sum().over("s1_id").cast(pl.UInt16),
        s1_hi_name_n=pl.col("_hi").sum().over("s1_id").cast(pl.UInt16),
        cand_num_support=pl.when(hi & (tn != "")).then(pl.col("_hi").sum().over("s1_id", "t_num"))
        .otherwise(0).cast(pl.UInt16),
        other_s1_exact=(pl.col("cand_exact_s1").fill_null(0) - pl.col("self_exact").fill_null(0)).cast(pl.UInt16),
    ).with_columns(
        twin_flag=(hi & both & (qn != tn) & (pl.col("s1_num_support") >= 1)).cast(pl.UInt8),
    )
    # v2 raw-address / name-difference features
    qt, tt = pl.col("q_numtoks"), pl.col("t_numtoks")
    has_q, has_t = qt.list.len() > 0, tt.list.len() > 0
    inter = qt.list.set_intersection(tt).list.len()
    union = qt.list.set_union(tt).list.len()
    comp = r"[-/]"
    q_comp = qt.list.eval(pl.element().filter(pl.element().str.contains(comp)))
    t_comp = tt.list.eval(pl.element().filter(pl.element().str.contains(comp)))
    qu, tu = pl.col("q_unit"), pl.col("t_unit")
    x = x.with_columns(
        numtok_jacc=pl.when(has_q & has_t).then(inter / union).otherwise(None).cast(pl.Float32),
        numtok_q_only=pl.when(has_t).then(qt.list.set_difference(tt).list.len()).otherwise(None).cast(pl.Float32),
        numtok_t_only=pl.when(has_q).then(tt.list.set_difference(qt).list.len()).otherwise(None).cast(pl.Float32),
        compound_shared=(q_comp.list.set_intersection(t_comp).list.len() > 0).cast(pl.UInt8),
        compound_conflict=((q_comp.list.len() > 0) & (t_comp.list.len() > 0)
                           & (q_comp.list.set_intersection(t_comp).list.len() == 0)).cast(pl.UInt8),
        unit_eq=(qu.is_not_null() & (qu == tu)).fill_null(False).cast(pl.UInt8),
        unit_conflict=(qu.is_not_null() & tu.is_not_null() & (qu != tu)).fill_null(False).cast(pl.UInt8),
        # informative name tokens present on only one side (an extra discriminating word)
        name_q_only=pl.col("q_ntoks").list.set_difference(pl.col("t_ntoks")).list.len().cast(pl.UInt8),
        name_t_only=pl.col("t_ntoks").list.set_difference(pl.col("q_ntoks")).list.len().cast(pl.UInt8),
    )
    keep = ID_COLS + ["num_absdiff", "num_reldiff", "num_prefix", "num_len_diff", "num_digit_lev",
                      "form_q_n", "form_t_n", "form_added", "form_dropped", "form_disjoint",
                      "s1_num_support", "s1_hi_name_n", "cand_num_support", "other_s1_exact", "twin_flag",
                      "numtok_jacc", "numtok_q_only", "numtok_t_only", "compound_shared", "compound_conflict",
                      "unit_eq", "unit_conflict", "name_q_only", "name_t_only"]
    return x.select(keep)


def main() -> None:
    """Build extra-feature shards for every country of a split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--countries", default="")
    a = ap.parse_args()
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    for country in (a.countries.split(",") if a.countries else countries_of(a.split)):
        t0 = time.time()
        f = build_extra(a.split, country)
        f.write_parquet(EXTRA_DIR / f"{a.split}_{country}.parquet")
        print(f"[{a.split}/{country}] extra rows={f.height:,} cols={f.width} secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
