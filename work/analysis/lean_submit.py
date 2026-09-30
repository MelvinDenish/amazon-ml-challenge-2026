"""Low-RAM re-selection of an existing test prediction (laptop: the full 46M-pair frame does not fit).

    python lean_submit.py <pred_parquet> <version> <reference_candidate_tsv> [--owner-prob]

Selection runs per country from a lazy scan (records never match across countries, so
this equals predict.py's global selection). The candidate file is copied from the run
that produced the predictions (identical candidate set), then the official validator
runs and a log row is appended, exactly as submission.write_submission does.
"""
import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from config import DATA_DIR  # noqa: E402
from io_utils import load_records, write_id_list_tsv  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from submission import SUBMISSION_DIR, VALIDATOR, sha1  # noqa: E402

pred_path, version, ref_cand = sys.argv[1], sys.argv[2], sys.argv[3]
own = "--owner-prob" in sys.argv
name = os.path.basename(pred_path).removeprefix("test_").removesuffix(".parquet")
parts = []
for c in ("France", "India", "US"):
    t = (pl.scan_parquet(pred_path).filter(pl.col("country") == c).collect()
         .rename({name: "p"}, strict=False).select("s1_id", "cand_id", "p"))
    parts.append(select_expected_f05(one_owner(exclusive_owner_prob(t, "p") if own else t, "p"), "p").select("s1_id", "cand_id"))
    print(c, "pairs", t.height, "selected", parts[-1].height, flush=True)
    del t
matches = pl.concat(parts)
out = SUBMISSION_DIR / version
out.mkdir(parents=True, exist_ok=True)
m_path, c_path = out / "matching_results.tsv", out / "candidate_pairs.tsv"
write_id_list_tsv(load_records("test", "source1")["entity_id"], matches, m_path, "matched_entity_ids")
shutil.copy(ref_cand, c_path)
res = subprocess.run([sys.executable, "-X", "utf8", str(VALIDATOR), "--matching", str(m_path), "--candidate", str(c_path),
                      "--test-dir", str(DATA_DIR / "test"), "--check-ids"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace",
                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print((res.stdout or "")[-1500:], (res.stderr or "")[-800:])
passed = res.returncode == 0
with open(SUBMISSION_DIR / "submission_log.csv", "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([version, datetime.now().isoformat(timespec="seconds"), sha1(m_path),
                            "PASS" if passed else "FAIL", matches.height, matches["s1_id"].n_unique(), "", "",
                            f"{name} ef05 owner_prob={own} (lean_submit)"])
print(f"{version}: validator {'PASS' if passed else 'FAIL'} -> {m_path}")
