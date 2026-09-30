"""Paths and global constants for the business entity resolution pipeline.

All paths can be overridden with environment variables so the pipeline runs
unchanged on a teammate's laptop or on Colab/Kaggle:

    BER_DATA_DIR       folder containing train/ and test/ (the raw TSVs)
    BER_ARTIFACT_DIR   folder for parquet copies, candidates, features, models
    BER_OUTPUT_DIR     folder receiving matching_results.tsv / candidate_pairs.tsv
"""

import os
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_WORK = _SRC.parents[2]  # .../work

DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR",
    _WORK.parent / "student_resource" / "student_resource" / "dataset",
))
ARTIFACT_DIR = Path(os.environ.get("BER_ARTIFACT_DIR", _WORK / "artifacts"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", _WORK / "output"))

PARQUET_DIR = ARTIFACT_DIR / "data"

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")
RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Test-set country shares, used to weight per-country validation scores.
# France has no labels; its weight is carried by a proxy (see score.py).
TEST_COUNTRY_WEIGHTS = {"US": 0.38, "India": 0.47, "France": 0.15}

SEED = 42

# XGBoost device: "cuda" on the laptop / Kaggle GPU; set BER_DEVICE=cpu on hosts
# without CUDA (e.g. the 96-core Kaggle TPU VM used for the CPU-bound steps).
DEVICE = os.environ.get("BER_DEVICE", "cuda")
