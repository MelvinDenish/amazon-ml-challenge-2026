"""Turn scored candidate pairs into per-S1 match sets, aligned with macro F0.5.

1. one_owner: every S2/S3 record belongs to at most one S1 in the ground truth,
   so each candidate record is kept only for its highest-scoring S1.
2. select_expected_f05: per S1, sort candidate probabilities p_1 >= p_2 >= ...
   and pick the prefix size m maximising the expected F0.5:
       m = 0 : E[F] = prod(1 - p_i)            (correct singleton)
       m > 0 : E[F] ~ 1.25 * sum_{i<=m} p_i / (0.25 * E|T| + m)
   with E|T| = sum_i p_i / recall_ceiling (true matches outside the candidate
   set are still counted in |truth|).
"""

import polars as pl


def ranked(df: pl.DataFrame, score: str, group: str, tiebreak: str, name: str) -> pl.DataFrame:
    """Add a deterministic 1-based rank of ``score`` (descending) within ``group``.

    Ties are broken by ``tiebreak`` ascending, so the same inputs always give the
    same ranks regardless of join/group-by row order (reproducible across machines).
    """
    return (
        df.sort([group, score, tiebreak], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).over(group).alias(name))
    )


def one_owner(pairs: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Keep each cand_id only for the S1 with its highest score (ties -> smallest s1_id)."""
    return ranked(pairs, score_col, "cand_id", "s1_id", "_own").filter(pl.col("_own") == 1).drop("_own")


def exclusive_owner_prob(pairs: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Turn the scores of all S1s competing for one record into owner probabilities.

    A record has at most one owner (or none), so with odds o_s = p_s / (1 - p_s):
        q_s = o_s / (1 + sum_s' o_s')      (the 1 is the null owner)
    The argmax owner is unchanged; contested pairs get lower probabilities before set
    selection. Measured on the v06 OOF: +0.0002 macro F0.5 in both India and US.
    """
    o = pl.col(score_col).cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    o = o / (1 - o)
    return pairs.with_columns((o / (1 + o.sum().over("cand_id"))).cast(pl.Float32).alias(score_col))


def select_threshold(pairs: pl.DataFrame, score_col: str, thr: float) -> pl.DataFrame:
    """Keep pairs whose score is at least ``thr``."""
    return pairs.filter(pl.col(score_col) >= thr)


def select_expected_f05(
    pairs: pl.DataFrame,
    prob_col: str = "p",
    recall_ceiling: float = 0.985,
    max_matches: int = 12,
) -> pl.DataFrame:
    """Per-S1 set selection maximising expected F0.5 over calibrated probabilities."""
    eps = 1e-6
    df = (
        pairs.sort(["s1_id", prob_col, "cand_id"], descending=[False, True, False])
        .with_columns(
            m=pl.col(prob_col).cum_count().over("s1_id"),
            cs=pl.col(prob_col).cum_sum().over("s1_id"),
            et=pl.col(prob_col).sum().over("s1_id") / recall_ceiling,
            ef0=(1.0 - pl.col(prob_col)).clip(eps, 1.0).log().sum().over("s1_id").exp(),
        )
        .with_columns(ef=1.25 * pl.col("cs") / (0.25 * pl.col("et") + pl.col("m")))
        .with_columns(
            ef=pl.when(pl.col("m") <= max_matches).then(pl.col("ef")).otherwise(-1.0)
        )
    )
    best = df.group_by("s1_id").agg(
        pl.col("ef").max().alias("ef_best"),
        pl.col("m").get(pl.col("ef").arg_max()).alias("m_best"),
        pl.col("ef0").first(),
    )
    df = df.join(best, on="s1_id")
    keep = (pl.col("ef_best") > pl.col("ef0")) & (pl.col("m") <= pl.col("m_best"))
    return df.filter(keep).drop("m", "cs", "et", "ef0", "ef", "ef_best", "m_best")
