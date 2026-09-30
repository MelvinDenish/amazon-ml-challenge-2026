"""Standard measurement report for a model run (the decision gate for every change).

    python diagnose.py --name xgb_v3 [--name xgb_v4x ...]

For each model:
  * train OOF macro F0.5 per country + test-mix estimate   (real labels)
  * false merges (FP) and missed true pairs (FN) per 1k S1 by error group:
      form conflict, near-number twin (|diff|<=10), other different-number,
      empty candidate address, rest                        (real labels)
  * TEST, label-free: selected pairs per 1k S1 in the same suspicious groups,
    and mean matches per S1, per country. On train those groups are mostly
    non-matches, so a fix should move the test counts toward train levels.
"""

import argparse

import polars as pl

from config import ARTIFACT_DIR
from features import FEAT_DIR, ID_COLS
from io_utils import load_ground_truth_pairs, load_records
from postprocess import one_owner, select_expected_f05
from score import report

EXTRA_DIR = ARTIFACT_DIR / "feats_extra"
OOF_DIR = ARTIFACT_DIR / "oof"


def _groups(split: str, countries: list[str]) -> pl.DataFrame:
    """Error-group label per pair from the base + extra feature shards."""
    parts = []
    for c in countries:
        base = pl.read_parquet(FEAT_DIR / f"{split}_{c}.parquet", columns=ID_COLS + ["t_a_empty"])
        ext = pl.read_parquet(EXTRA_DIR / f"{split}_{c}.parquet",
                              columns=ID_COLS + ["form_disjoint", "twin_flag", "num_absdiff"])
        parts.append(base.join(ext, on=ID_COLS))
    g = pl.concat(parts)
    return g.select(ID_COLS + [
        pl.when(pl.col("form_disjoint") == 1).then(pl.lit("form conflict"))
        .when((pl.col("twin_flag") == 1) & (pl.col("num_absdiff") <= 10)).then(pl.lit("near-number twin"))
        .when(pl.col("twin_flag") == 1).then(pl.lit("other-number twin"))
        .when(pl.col("t_a_empty") == 1).then(pl.lit("empty cand address"))
        .otherwise(pl.lit("rest")).alias("group")
    ])


def diagnose(name: str, test_only: bool = False) -> None:
    """Print the train-OOF and test-side diagnostics for one model.

    ``test_only`` skips the train part (for hosts without the train feature shards).
    """
    if test_only:
        _test_side(name)
        return
    oof = pl.read_parquet(OOF_DIR / f"{name}.parquet")
    countries = sorted(oof["country"].unique().to_list())
    s1 = load_records("train", "source1").join(
        oof.select(pl.col("s1_id").alias("entity_id")).unique(), on="entity_id", how="semi")
    gt = load_ground_truth_pairs().join(oof.select("s1_id").unique(), on="s1_id", how="semi")
    sel = select_expected_f05(one_owner(oof, "p"), "p")
    print(f"\n######## {name}")
    print("TRAIN OOF:", {k: round(v, 4) for k, v in report(gt, sel, s1).items()})

    g = _groups("train", countries)
    n = oof["s1_id"].n_unique()
    fp = sel.join(gt, on=ID_COLS, how="anti").join(g, on=ID_COLS, how="left")
    fn = oof.filter(pl.col("y") == 1).join(sel, on=ID_COLS, how="anti").join(g, on=ID_COLS, how="left")
    tab = (fp.group_by("group").agg((pl.len() * 1000 / n).round(2).alias("FP_per_1k"))
           .join(fn.group_by("group").agg((pl.len() * 1000 / n).round(2).alias("FN_per_1k")), on="group", how="full",
                 coalesce=True).sort("group"))
    print("TRAIN errors by group (per 1k S1):", tab.rows())

    _test_side(name)
    trs = sel.join(g, on=ID_COLS, how="left").group_by("group").len("k").with_columns(
        (pl.col("k") * 1000 / n).round(1).alias("per_1k")).select("group", "per_1k").sort("group")
    print("TRAIN selected pairs per 1k S1 by group (reference):", trs.rows())


def _test_side(name: str) -> None:
    """Label-free test diagnostics: selected pairs per 1k S1 by suspicious group."""
    tpath = OOF_DIR / f"test_{name}.parquet"
    if not tpath.exists():
        print("TEST: no test predictions yet")
        return
    t = pl.read_parquet(tpath)
    tsel = select_expected_f05(one_owner(t, "p"), "p")
    tg = _groups("test", sorted(t["country"].unique().to_list()))
    ts = tsel.join(tg, on=ID_COLS, how="left")
    n_s1 = load_records("test", "source1").group_by("country").len("n_s1")
    per = (ts.group_by("country", "group").len("k").join(n_s1, on="country")
           .with_columns((pl.col("k") * 1000 / pl.col("n_s1")).round(1).alias("per_1k"))
           .pivot(on="group", index="country", values="per_1k").sort("country"))
    mm = ts.group_by("country").len("pairs").join(n_s1, on="country").with_columns(
        (pl.col("pairs") / pl.col("n_s1")).round(3).alias("matches_per_S1")).select("country", "matches_per_S1")
    print("TEST selected pairs per 1k S1 by group:")
    print(per.join(mm, on="country"))


if __name__ == "__main__":
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--test-only", action="store_true", help="skip the train-OOF part")
    args = ap.parse_args()
    for nm in args.name:
        diagnose(nm, test_only=args.test_only)
