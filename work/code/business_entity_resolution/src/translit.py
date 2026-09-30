"""Native-script -> Latin token map learned ONLY from the training ground truth.

About 11% of S2/S3 names (and many state names in addresses) are written in
Indic scripts, while S1 is Latin. No transliteration library or external data
is used: we learn token translations from matched training pairs.

  names      S1 name and native S2/S3 name with the same token count are
             aligned position by position ("राम मार्केटिंग प्राइवेट लिमिटेड" ->
             "ram marketing private limited").
  addresses  each native address token is mapped to the S1 address token it
             co-occurs with most often (state names such as "पश्चिमबंग").

A mapping is kept when seen >= MIN_COUNT times and its share among the native
token's alignments is >= MIN_SHARE. ``apply_map`` rewrites native tokens and
leaves unknown ones untouched.

    python translit.py   # learn from train, save artifacts/translit_map.parquet, report coverage
"""

import polars as pl

from config import ARTIFACT_DIR
from io_utils import load_ground_truth_pairs, load_records
from normalize import normalize_records

MAP_PATH = ARTIFACT_DIR / "translit_map.parquet"
MIN_COUNT = 2
MIN_SHARE = 0.5
NATIVE_RE = (r"[\p{Devanagari}\p{Bengali}\p{Kannada}\p{Tamil}\p{Telugu}\p{Gujarati}"
             r"\p{Gurmukhi}\p{Malayalam}\p{Oriya}]")


def _tokens(col: str) -> pl.Expr:
    """Split a cleaned text column into a list of non-empty tokens."""
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _best(pairs: pl.DataFrame, min_share: float = MIN_SHARE) -> pl.DataFrame:
    """Keep each native token's dominant latin token.

    ``pairs`` has one row per (row id ``r``, native, latin) alignment; the share
    is the fraction of rows containing the native token that align it to latin.
    """
    cnt = pairs.group_by("native", "latin").agg(pl.col("r").n_unique().alias("n"))
    tot = pairs.group_by("native").agg(pl.col("r").n_unique().alias("tot"))
    return (
        cnt.join(tot, on="native")
        .with_columns(share=pl.col("n") / pl.col("tot"))
        .filter((pl.col("n") >= MIN_COUNT) & (pl.col("share") >= min_share))
        .sort(["n", "latin"], descending=[True, False])  # deterministic tie-break
        .unique("native", keep="first")
        .select("native", "latin")
    )


def learn_map() -> pl.DataFrame:
    """Learn the native->latin token map from train (India) matched pairs."""
    gt = load_ground_truth_pairs()
    s1 = normalize_records(load_records("train", "source1", "India")).select(
        pl.col("entity_id").alias("s1_id"), pl.col("n_full").alias("s_name"), pl.col("a_norm").alias("s_addr"))
    t = pl.concat([load_records("train", "source2", "India"), load_records("train", "source3", "India")])
    t = normalize_records(t).select(
        pl.col("entity_id").alias("cand_id"), pl.col("n_full").alias("t_name"), pl.col("a_norm").alias("t_addr"))
    pairs = gt.join(s1, on="s1_id").join(t, on="cand_id")

    # Names: positional alignment when token counts agree.
    nm = pairs.filter(pl.col("t_name").str.contains(NATIVE_RE)).select(
        _tokens("s_name").alias("l"), _tokens("t_name").alias("n"))
    nm = nm.filter(pl.col("l").list.len() == pl.col("n").list.len()).with_row_index("r")
    nm = nm.explode("l", "n").rename({"l": "latin", "n": "native"})
    nm = nm.filter(pl.col("native").str.contains(NATIVE_RE) & ~pl.col("latin").str.contains(NATIVE_RE))
    name_map = _best(nm)

    # Addresses: co-occurrence of native tokens with S1 address tokens.
    ad = pairs.filter(pl.col("t_addr").str.contains(NATIVE_RE)).select(
        _tokens("s_addr").list.unique().alias("latin"),
        _tokens("t_addr").list.eval(pl.element().filter(pl.element().str.contains(NATIVE_RE))).alias("native"),
    ).with_row_index("r").explode("native").explode("latin").drop_nulls()
    ad = ad.filter(~pl.col("latin").str.contains(r"\d") & (pl.col("latin").str.len_chars() >= 2))
    addr_map = _best(ad.select("r", "native", "latin"), min_share=0.8)

    m = pl.concat([name_map, addr_map.join(name_map, on="native", how="anti")])
    print(f"learned {name_map.height:,} name tokens, {addr_map.height:,} address tokens")
    return m


def apply_map(df: pl.DataFrame, mapping: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Replace native tokens in the given space-separated text columns."""
    natives, latins = mapping["native"].to_list(), mapping["latin"].to_list()
    return df.with_columns([
        _tokens(c).list.eval(pl.element().replace(natives, latins)).list.join(" ").alias(c)
        for c in cols
    ])


def coverage(mapping: pl.DataFrame, split: str = "test") -> float:
    """Share of native name tokens (by occurrence) of a split that the map covers."""
    t = pl.concat([load_records(split, "source2", "India"), load_records(split, "source3", "India")])
    toks = normalize_records(t).select(_tokens("n_full").alias("x")).explode("x")
    toks = toks.filter(pl.col("x").str.contains(NATIVE_RE))
    return float(toks["x"].is_in(mapping["native"].implode()).mean())


if __name__ == "__main__":
    mp = learn_map()
    mp.write_parquet(MAP_PATH)
    print("sample:", mp.sample(15, seed=1).rows())
    print(f"test native-token coverage: {coverage(mp):.3f}")
