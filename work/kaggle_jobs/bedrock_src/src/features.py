"""Pairwise features for (S1, S2/S3) candidate pairs.

    python features.py --split train --frac 0.4   # sample of train S1 (by S1 id)
    python features.py --split test

Feature groups
  name     rapidfuzz ratios on cleaned names (core, full, no-space, alias part),
           Jaro-Winkler, token Jaccard, token counts, record flags
  address  rapidfuzz ratios, house-number / street / PIN agreement, token Jaccard
  context  blocking score/bits, rank of the pair among the S1's candidates and
           among the candidate's S1s, margins to the best competitor, number of
           competitors, name/address "genericness" counts
Country is deliberately NOT a feature (open set; France unseen in train).
"""

import argparse
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from build_candidates import CAND_DIR, countries_of
from config import ARTIFACT_DIR, SEED
from dataset import load_country
from postprocess import ranked

FEAT_DIR = ARTIFACT_DIR / "feats"
ID_COLS = ["s1_id", "cand_id"]


def _cp(a: list, b: list, scorer) -> np.ndarray:
    """Vectorised pairwise similarity scaled to [0, 1] as float32."""
    s = process.cpdist(a, b, scorer=scorer, workers=-1)
    s = s.astype("float32")
    return s / 100.0 if s.max(initial=0) > 1.0 else s


def _jaccard(a: str, b: str) -> pl.Expr:
    """Jaccard similarity of two list columns (0 when both are empty)."""
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0).cast(pl.Float32)


def context_features(c: pl.DataFrame) -> pl.DataFrame:
    """Rank / competition features computed over the whole candidate table."""
    c = ranked(c, "rscore", "cand_id", "s1_id", "rev_rank")  # deterministic tie-break
    return c.with_columns(
        n_cands_s1=pl.len().over("s1_id").cast(pl.UInt16),
        n_s1_cand=pl.len().over("cand_id").cast(pl.UInt16),
        rev_rank=pl.col("rev_rank").cast(pl.UInt16),
        s1_max_rscore=pl.col("rscore").max().over("s1_id"),
        cand_max_rscore=pl.col("rscore").max().over("cand_id"),
        n_strong_s1=(pl.col("rscore") >= 1.7).sum().over("s1_id").cast(pl.UInt16),
        n_strong_cand=(pl.col("rscore") >= 1.7).sum().over("cand_id").cast(pl.UInt16),
    ).with_columns(
        margin_s1=(pl.col("rscore") - pl.col("s1_max_rscore")).cast(pl.Float32),
        margin_cand=(pl.col("rscore") - pl.col("cand_max_rscore")).cast(pl.Float32),
        # best competing S1 for this candidate (excluding this pair when it is the best)
        cand_second=pl.col("rscore").sort(descending=True).slice(1, 1).first().over("cand_id"),
    ).with_columns(
        margin_cand2=(pl.col("rscore") - pl.col("cand_second").fill_null(0.0)).cast(pl.Float32),
    ).drop("cand_second")


def pair_features(c: pl.DataFrame, q: pl.DataFrame, t: pl.DataFrame, sub_chunk: int = 2_000_000) -> pl.DataFrame:
    """String-similarity features for candidate pairs of one country."""
    qcols = ["entity_id", "n_core", "n_full", "n_nospace", "n_toks", "a_norm", "a_toks", "a_num",
             "a_street", "n_core_cnt", "a_norm_cnt"]
    tcols = qcols + ["n_alias", "is_native", "was_native", "is_domain", "has_alias", "a_empty"]
    qx = q.with_columns(
        n_core_cnt=pl.len().over("n_core").cast(pl.UInt32),
        a_norm_cnt=pl.len().over("a_norm").cast(pl.UInt32),
    ).select(qcols).rename(lambda x: "q_" + x)
    tx = t.with_columns(
        n_core_cnt=pl.len().over("n_core").cast(pl.UInt32),
        a_norm_cnt=pl.len().over("a_norm").cast(pl.UInt32),
    ).select(tcols).rename(lambda x: "t_" + x)

    out = []
    for s in range(0, c.height, sub_chunk):
        p = (
            c.slice(s, sub_chunk)
            .join(qx, left_on="s1_id", right_on="q_entity_id", how="left", maintain_order="left")
            .join(tx, left_on="cand_id", right_on="t_entity_id", how="left", maintain_order="left")
        )
        qn, tn = p["q_n_core"].to_list(), p["t_n_core"].to_list()
        qa, ta = p["q_a_norm"].to_list(), p["t_a_norm"].to_list()
        qs, ts = p["q_n_nospace"].to_list(), p["t_n_nospace"].to_list()
        f = {
            "n_tset": _cp(qn, tn, fuzz.token_set_ratio),
            "n_tsort": _cp(qn, tn, fuzz.token_sort_ratio),
            "n_ratio": _cp(qn, tn, fuzz.ratio),
            "n_partial": _cp(qn, tn, fuzz.partial_ratio),
            "n_full_tset": _cp(qn, p["t_n_full"].to_list(), fuzz.token_set_ratio),
            "n_alias_tset": _cp(qn, p["t_n_alias"].to_list(), fuzz.token_set_ratio),
            "n_nospace_ratio": _cp(qs, ts, fuzz.ratio),
            "n_nospace_partial": _cp(qs, ts, fuzz.partial_ratio),
            "n_jw": _cp(qs, ts, JaroWinkler.normalized_similarity),
            "a_tset": _cp(qa, ta, fuzz.token_set_ratio),
            "a_tsort": _cp(qa, ta, fuzz.token_sort_ratio),
            "a_partial": _cp(qa, ta, fuzz.partial_ratio),
            "a_ratio": _cp(qa, ta, fuzz.ratio),
        }
        p = p.with_columns([pl.Series(k, v) for k, v in f.items()])
        pin = r"\b(\d{6})\b"
        p = p.with_columns(
            n_jacc=_jaccard("q_n_toks", "t_n_toks"),
            a_jacc=_jaccard("q_a_toks", "t_a_toks"),
            q_n_ntok=pl.col("q_n_toks").list.len().cast(pl.UInt8),
            t_n_ntok=pl.col("t_n_toks").list.len().cast(pl.UInt8),
            t_a_ntok=pl.col("t_a_toks").list.len().cast(pl.UInt8),
            num_both=((pl.col("q_a_num") != "") & (pl.col("t_a_num") != "")),
            num_eq=((pl.col("q_a_num") != "") & (pl.col("q_a_num") == pl.col("t_a_num"))),
            street_eq=((pl.col("q_a_street") != "") & (pl.col("q_a_street") == pl.col("t_a_street"))),
            num_in_t=pl.col("t_a_toks").list.contains(pl.col("q_a_num")),
            _qpin=pl.col("q_a_norm").str.extract(pin),
            _tpin=pl.col("t_a_norm").str.extract(pin),
            is_s3=pl.col("cand_id").str.starts_with("S3-"),
        ).with_columns(
            pin_eq=(pl.col("_qpin").is_not_null() & (pl.col("_qpin") == pl.col("_tpin"))).fill_null(False),
            pin_conflict=(pl.col("_qpin").is_not_null() & pl.col("_tpin").is_not_null()
                          & (pl.col("_qpin") != pl.col("_tpin"))).fill_null(False),
            num_conflict=pl.col("num_both") & ~pl.col("num_eq"),
        )
        drop = [x for x in p.columns if x.startswith("q_") and x not in
                ("q_n_ntok", "q_n_core_cnt", "q_a_norm_cnt")]
        drop += [x for x in p.columns if x.startswith("t_") and x not in
                 ("t_n_ntok", "t_a_ntok", "t_n_core_cnt", "t_a_norm_cnt", "t_is_native", "t_was_native",
                  "t_is_domain", "t_has_alias", "t_a_empty")]
        out.append(p.drop(drop + ["_qpin", "_tpin", "t_native", "t_aempty"], strict=False))
    return pl.concat(out)


def build_features(split: str, country: str, frac: float = 1.0) -> pl.DataFrame:
    """Load candidates for a split/country, compute all features, optionally sample S1."""
    c = pl.read_parquet(CAND_DIR / f"{split}_{country}.parquet")
    c = context_features(c)  # computed on the full candidate table, before sampling
    if frac < 1.0:
        # Deterministic hash rule (not unique().sample(): unique() order is not stable,
        # so two runs sampled different S1s and OOF blends silently shrank to the overlap).
        c = c.filter((pl.col("s1_id").hash(seed=SEED + 3) % 10_000) < int(frac * 10_000))
    q, t = load_country(split, country)
    feats = pair_features(c, q, t)
    bools = [k for k, v in feats.schema.items() if v == pl.Boolean]
    return feats.with_columns(pl.col(bools).cast(pl.UInt8))


def main() -> None:
    """Build and save feature shards for the requested split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--countries", default="")
    a = ap.parse_args()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    for country in (a.countries.split(",") if a.countries else countries_of(a.split)):
        t0 = time.time()
        f = build_features(a.split, country, a.frac)
        f.write_parquet(FEAT_DIR / f"{a.split}_{country}.parquet")
        print(f"[{a.split}/{country}] rows={f.height:,} cols={f.width} secs={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
