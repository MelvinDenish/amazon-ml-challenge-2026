"""Load one country's S1 (queries) and S2+S3 (targets) as normalised frames."""

import polars as pl

from config import ARTIFACT_DIR
from io_utils import load_ground_truth_pairs, load_records
from normalize import normalize_records, transliterate

TRANSLIT_MAP = ARTIFACT_DIR / "translit_map.parquet"


def load_country(split: str, country: str, translit: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (q, t) for a split and country.

    q: S1 records with integer index ``q``; t: S2+S3 records with index ``t``.
    Both carry all normalised columns from normalize.normalize_records; native
    script tokens in t are mapped to Latin with the train-learned map (translit.py).
    """
    q = load_records(split, "source1", country)
    t = pl.concat([load_records(split, "source2", country), load_records(split, "source3", country)])
    q = normalize_records(q)
    t = normalize_records(t)
    if translit and TRANSLIT_MAP.exists():
        mapping = pl.read_parquet(TRANSLIT_MAP)
        q = transliterate(q, mapping)
        t = transliterate(t, mapping)
    return q.with_row_index("q"), t.with_row_index("t")


def truth_in_index_space(q: pl.DataFrame, t: pl.DataFrame) -> pl.DataFrame:
    """Return train ground-truth matches of these records as (q, t) index pairs."""
    gt = load_ground_truth_pairs()
    return (
        gt.join(q.select(pl.col("entity_id").alias("s1_id"), "q"), on="s1_id")
        .join(t.select(pl.col("entity_id").alias("cand_id"), "t"), on="cand_id")
        .select("q", "t")
    )
