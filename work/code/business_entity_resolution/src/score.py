"""Exact re-implementation of the challenge metric: macro F0.5 over Source-1 entities.

Per S1 entity:  F0.5 = 1.25*TP / (0.25*|truth| + |pred|)
  * truth empty and pred empty  -> 1.0  (correct singleton)
  * exactly one of them empty   -> 0.0
The final score is the mean over ALL S1 entities in the evaluation set.
"""

import polars as pl

from config import TEST_COUNTRY_WEIGHTS


def per_entity_f05(truth: pl.DataFrame, pred: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    """Return one row per S1 id with columns (s1_id, n_true, n_pred, tp, f05).

    ``truth`` and ``pred`` are pair tables with columns (s1_id, cand_id).
    Only S1 ids listed in ``s1_ids`` are scored; pairs for other ids are ignored.
    """
    base = pl.DataFrame({"s1_id": s1_ids}).unique()
    t = truth.select("s1_id", "cand_id").unique()
    p = pred.select("s1_id", "cand_id").unique()
    n_true = t.group_by("s1_id").len("n_true")
    n_pred = p.group_by("s1_id").len("n_pred")
    tp = t.join(p, on=["s1_id", "cand_id"]).group_by("s1_id").len("tp")
    df = (
        base.join(n_true, on="s1_id", how="left")
        .join(n_pred, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0))
    )
    return df.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .when((pl.col("n_true") == 0) | (pl.col("n_pred") == 0)).then(0.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")))
        .alias("f05")
    )


def macro_f05(truth: pl.DataFrame, pred: pl.DataFrame, s1_ids: pl.Series) -> float:
    """Macro-averaged F0.5 over ``s1_ids`` (the leaderboard metric)."""
    return float(per_entity_f05(truth, pred, s1_ids)["f05"].mean())


def report(truth: pl.DataFrame, pred: pl.DataFrame, s1: pl.DataFrame) -> dict:
    """Score per country and a test-weighted mean.

    ``s1`` has columns (entity_id, country). France has no labels, so its test
    weight is carried by min(US, India) as a pessimistic proxy.
    """
    scores = per_entity_f05(truth, pred, s1["entity_id"]).join(
        s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id"
    )
    by_country = {
        r["country"]: r["f05"] for r in scores.group_by("country").agg(pl.col("f05").mean()).to_dicts()
    }
    out = {"overall": float(scores["f05"].mean()), **{k: round(v, 5) for k, v in by_country.items()}}
    if "US" in by_country and "India" in by_country:
        w = TEST_COUNTRY_WEIGHTS
        out["test_weighted"] = (
            w["US"] * by_country["US"] + w["India"] * by_country["India"]
            + w["France"] * min(by_country["US"], by_country["India"])
        )
    return out


def _self_test() -> None:
    """Check the implementation against the README example and edge cases."""
    truth = pl.DataFrame({"s1_id": ["A", "A", "C"], "cand_id": ["S2-47", "S3-812", "S3-9"]})
    pred = pl.DataFrame({"s1_id": ["A", "A", "A", "D"], "cand_id": ["S2-47", "S2-193", "S3-812", "S2-1"]})
    ids = pl.Series(["A", "B", "C", "D"])
    f = {r["s1_id"]: r["f05"] for r in per_entity_f05(truth, pred, ids).to_dicts()}
    assert abs(f["A"] - 0.714285) < 1e-4, f["A"]  # README example
    assert f["B"] == 1.0  # correct singleton
    assert f["C"] == 0.0  # missed all matches
    assert f["D"] == 0.0  # false merge on a singleton
    print("score.py self-test passed:", f)


if __name__ == "__main__":
    _self_test()
