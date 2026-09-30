"""Build and cache blocking candidates for every (split, country).

Usage:  python build_candidates.py [--split train|test|all] [--countries US,India]

Writes artifacts/cands/<split>_<country>.parquet with one row per candidate
pair: s1_id, cand_id, bscore, kbits, sim_n, sim_a, t_native, t_aempty, rscore,
brank. For train it also prints pair / full-cluster recall against the ground
truth.
"""

import argparse
import gc
import time

import polars as pl

from blocking import blocking_recall, generate_candidates
from config import ARTIFACT_DIR
from dataset import load_country, truth_in_index_space
from io_utils import load_records

CAND_DIR = ARTIFACT_DIR / "cands"
# pool_k 600 (was 150): measured on train, the stage-1 pool cut was the main
# blocking loss (India recall +1.45 pts, US +0.71 at the same top-20).
BLOCKING_PARAMS = {"top_k": 20, "pool_k": 600, "top_rev": 3, "chunk": 25_000}


def countries_of(split: str) -> list[str]:
    """Return the country labels present in a split's S1 file (open set)."""
    return sorted(load_records(split, "source1")["country"].unique().to_list())


def build(split: str, country: str) -> None:
    """Generate candidates for one split/country, save them, report recall on train."""
    t0 = time.time()
    q, t = load_country(split, country)
    # id maps and truth are computed up front so the large normalised frames can
    # be released before the memory-heavy final merge (US train OOM'd at 30 GB)
    q_ids = q.select("q", pl.col("entity_id").alias("s1_id"))
    t_ids = t.select("t", pl.col("entity_id").alias("cand_id"))
    truth = truth_in_index_space(q, t) if split == "train" else None
    n_q, n_t = q.height, t.height
    cands = generate_candidates(q, t, **BLOCKING_PARAMS)
    del q, t
    gc.collect()
    msg = f"[{split}/{country}] q={n_q:,} t={n_t:,} secs={time.time() - t0:.0f}"
    if split == "train":
        msg += f" {blocking_recall(cands, truth, n_q)}"
    else:
        msg += f" cands_per_s1={cands.height / n_q:.2f}"
    print(msg, flush=True)
    out = cands.join(q_ids, on="q").join(t_ids, on="t").drop("q", "t")
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(CAND_DIR / f"{split}_{country}.parquet")


def main() -> None:
    """Parse arguments and build candidates for the requested splits/countries."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--countries", default="")
    args = ap.parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        wanted = args.countries.split(",") if args.countries else countries_of(split)
        for country in wanted:
            build(split, country)


if __name__ == "__main__":
    main()
