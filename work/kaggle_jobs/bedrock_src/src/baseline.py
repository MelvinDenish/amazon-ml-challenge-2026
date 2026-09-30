"""Fail-safe rule baseline: threshold on the stage-2 blocking score.

    python baseline.py tune              # grid-search the threshold on train
    python baseline.py submit --thr 1.6  # write + validate a test submission

The rule: after global one-owner assignment on ``rscore`` (name + address
similarity with neutral values for native-script names / empty addresses),
keep every pair whose rscore >= thr.
"""

import argparse

import polars as pl

from build_candidates import CAND_DIR
from io_utils import load_ground_truth_pairs, load_records
from postprocess import one_owner, select_threshold
from score import report
from submission import write_submission


def load_cands(split: str, countries: list[str] | None = None) -> pl.DataFrame:
    """Load cached candidate parquet files of a split (all countries by default)."""
    files = sorted(CAND_DIR.glob(f"{split}_*.parquet"))
    if countries:
        files = [f for f in files if f.stem.split("_", 1)[1] in countries]
    return pl.concat([pl.read_parquet(f) for f in files])


def tune(countries: list[str]) -> None:
    """Print the macro F0.5 per country for a grid of thresholds on train."""
    cands = one_owner(load_cands("train", countries), "rscore")
    s1 = load_records("train", "source1").filter(pl.col("country").is_in(countries))
    truth = load_ground_truth_pairs().join(
        s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    for thr in [x / 100 for x in range(150, 246, 5)]:
        print(f"thr={thr:.2f}", report(truth, select_threshold(cands, "rscore", thr), s1), flush=True)


def submit(thr: float, version: str) -> None:
    """Apply the threshold rule to test candidates and write a submission."""
    cands = load_cands("test")
    matches = select_threshold(one_owner(cands, "rscore"), "rscore", thr)
    write_submission(version, matches, cands, notes=f"rule baseline rscore>={thr}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "submit"])
    ap.add_argument("--thr", type=float, default=1.6)
    ap.add_argument("--version", default="v01_rule_baseline")
    ap.add_argument("--countries", default="India,US")
    a = ap.parse_args()
    if a.mode == "tune":
        tune(a.countries.split(","))
    else:
        submit(a.thr, a.version)
