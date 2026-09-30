"""Reading raw TSVs, caching them as parquet, and writing submission files."""

from pathlib import Path

import polars as pl

from config import DATA_DIR, PARQUET_DIR, RECORD_COLUMNS, SOURCES, SPLITS


def raw_path(split: str, name: str) -> Path:
    """Return the raw TSV path, e.g. raw_path('train', 'source2')."""
    return DATA_DIR / split / f"{split}_{name}.tsv"


def read_raw_tsv(path: Path) -> pl.DataFrame:
    """Read a challenge TSV with every column as a string.

    Quoting is disabled because names contain apostrophes and quotes, and empty
    cells stay as empty strings (never null) so string ops need no null checks.
    """
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema_length=0,  # all columns Utf8
        missing_utf8_is_empty_string=True,
    )
    return df.with_columns(pl.all().fill_null(""))


def parquet_path(split: str, name: str) -> Path:
    """Return the cached parquet path for a split/source (or 'ground_truth')."""
    return PARQUET_DIR / f"{split}_{name}.parquet"


def convert_all_to_parquet(overwrite: bool = False) -> None:
    """Convert every raw TSV (sources and train ground truth) to parquet once."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(s, n) for s in SPLITS for n in SOURCES] + [("train", "ground_truth")]
    for split, name in jobs:
        out = parquet_path(split, name)
        if out.exists() and not overwrite:
            continue
        df = read_raw_tsv(raw_path(split, name))
        if name != "ground_truth":
            assert df.columns == RECORD_COLUMNS, df.columns
        df.write_parquet(out, compression="zstd")
        print(f"  wrote {out.name}: {df.height:,} rows")


def load_records(split: str, name: str, country: str | None = None) -> pl.DataFrame:
    """Load a source's records from parquet, optionally filtered to one country."""
    lf = pl.scan_parquet(parquet_path(split, name))
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    return lf.collect()


def load_ground_truth_pairs() -> pl.DataFrame:
    """Return train ground truth exploded to one (s1_id, cand_id) row per true match."""
    gt = pl.read_parquet(parquet_path("train", "ground_truth"))
    return (
        gt.filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
    )


def write_id_list_tsv(
    s1_ids: list[str] | pl.Series,
    pairs: pl.DataFrame,
    path: Path,
    list_column: str,
) -> None:
    """Write a submission-format TSV: one row per S1 id, comma-joined S2/S3 ids.

    ``pairs`` has columns (s1_id, cand_id). S1 ids with no pairs get an empty
    cell. Duplicate pairs are removed so the list never repeats an id.
    """
    grouped = (
        pairs.select("s1_id", "cand_id").unique()
        .sort("s1_id", "cand_id")
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("cand_id").str.join(","))
    )
    base = pl.DataFrame({"s1_id": pl.Series(s1_ids, dtype=pl.Utf8)}).unique(maintain_order=True)
    out = (
        base.join(grouped, on="s1_id", how="left")
        .with_columns(pl.col("cand_id").fill_null(""))
        .rename({"s1_id": "source1_entity_id", "cand_id": list_column})
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
