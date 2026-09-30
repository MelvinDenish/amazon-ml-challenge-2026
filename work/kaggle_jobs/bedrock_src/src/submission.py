"""Write both submission TSVs, validate them, and log the version.

Every leaderboard upload goes through ``write_submission`` so that:
  * both files follow the exact format (one row per test S1, tab-separated),
  * the official validator runs with --check-ids,
  * a row (version, time, sha1, CV, notes) is appended to submission_log.csv.
"""

import csv
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from config import ARTIFACT_DIR, DATA_DIR
from io_utils import load_records, write_id_list_tsv

SUBMISSION_DIR = ARTIFACT_DIR.parent / "submissions"
VALIDATOR = DATA_DIR.parent / "utils" / "validate_submission.py"


def sha1(path: Path) -> str:
    """Return the sha1 hex digest of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_submission(
    version: str,
    matches: pl.DataFrame,
    candidates: pl.DataFrame,
    cv: dict | None = None,
    notes: str = "",
    out_dir: Path | None = None,
) -> Path:
    """Write matching_results.tsv + candidate_pairs.tsv, validate, and log.

    ``matches`` and ``candidates`` are pair tables (s1_id, cand_id). Matches
    are forced to be a subset of candidates.
    """
    out = out_dir or SUBMISSION_DIR / version
    s1_ids = load_records("test", "source1")["entity_id"]
    matches = matches.select("s1_id", "cand_id").join(
        candidates.select("s1_id", "cand_id"), on=["s1_id", "cand_id"], how="semi"
    )
    m_path, c_path = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    write_id_list_tsv(s1_ids, matches, m_path, "matched_entity_ids")
    write_id_list_tsv(s1_ids, candidates, c_path, "candidate_entity_ids")

    res = subprocess.run(
        [sys.executable, "-X", "utf8", str(VALIDATOR), "--matching", str(m_path), "--candidate", str(c_path),
         "--test-dir", str(DATA_DIR / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    print((res.stdout or "")[-2000:], (res.stderr or "")[-1000:])
    passed = res.returncode == 0

    log = SUBMISSION_DIR / "submission_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["version", "time", "sha1", "validator", "n_matched_pairs",
                        "n_nonempty_s1", "cv", "lb", "notes"])
        w.writerow([
            version, datetime.now().isoformat(timespec="seconds"), sha1(m_path),
            "PASS" if passed else "FAIL", matches.height, matches["s1_id"].n_unique(),
            cv or "", "", notes,
        ])
    print(f"{version}: validator {'PASS' if passed else 'FAIL'} -> {m_path}")
    return m_path
