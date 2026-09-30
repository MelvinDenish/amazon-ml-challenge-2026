"""Candidate generation (blocking): multi-key inverted index within a country.

Stage 1: every record emits several hashed blocking keys:

  A   house number + street word             ("801|broad")
  N   each informative name token            ("velex")
  P   first 10 chars of the no-space name    ("lopezsmith"; catches domains/handles)
  S   sorted informative name tokens         ("campbell lopez smith")
  R   each rare alphabetic address token     ("belgachhi"; India landmarks/localities)
  B   adjacent address-token bigrams         ("panch batti")
  NB  unordered pairs of name tokens         ("high trading"; robust to word order)
  AN  house number + name token              ("801|regional")

Keys that are too common on the S2/S3 side are dropped (per-type caps). A
pair's stage-1 score is the sum over shared keys of weight(type)/log2(1+df);
the top ``pool_k`` targets per S1 form the stage-1 pool.

Stage 2 re-ranks the pool with real string similarity (rapidfuzz token-set
ratio on the cleaned name and address) and keeps the top-K per S1 plus the
top-M S1 per S2/S3 record (record-centric recall). That union is the exact
candidate set scored by the matcher and written to candidate_pairs.tsv.
"""

import polars as pl
from rapidfuzz import fuzz, process

from normalize import ADDR_STOP
from postprocess import ranked

KEY_CAPS = {"A": 400, "N": 300, "P": 300, "S": 300, "R": 200, "B": 300, "NB": 300, "AN": 300}
KEY_WEIGHTS = {"A": 3.0, "N": 1.0, "P": 1.5, "S": 2.0, "R": 1.0, "B": 1.5, "NB": 1.5, "AN": 2.0}
KEY_BITS = {"A": 1, "N": 2, "P": 4, "S": 8, "R": 16, "B": 32, "NB": 64, "AN": 128}


def record_keys(df: pl.DataFrame, idx_col: str) -> pl.DataFrame:
    """Return (idx, ktype, h) rows: one per blocking key of each record.

    ``df`` must be normalised (see normalize.normalize_records) and carry an
    integer row index column ``idx_col``.
    """
    addr_stop = list(ADDR_STOP)
    parts = []

    a = df.filter((pl.col("a_num") != "") & (pl.col("a_street") != "")).select(
        pl.col(idx_col), pl.lit("A").alias("ktype"),
        pl.concat_str(["a_num", "a_street"], separator="|").alias("k"),
    )
    parts.append(a)

    n = df.select(pl.col(idx_col), pl.col("n_toks").list.unique().alias("k")).explode("k")
    n = n.filter(pl.col("k").str.len_chars() >= 3).with_columns(pl.lit("N").alias("ktype"))
    parts.append(n.select(idx_col, "ktype", "k"))

    p = df.filter(pl.col("n_nospace").str.len_chars() >= 6).select(
        pl.col(idx_col), pl.lit("P").alias("ktype"), pl.col("n_nospace").str.slice(0, 10).alias("k")
    )
    parts.append(p)

    s = df.filter(pl.col("n_toks").list.len() >= 1).select(
        pl.col(idx_col), pl.lit("S").alias("ktype"),
        pl.col("n_toks").list.sort().list.join(" ").alias("k"),
    )
    parts.append(s)

    r = df.select(pl.col(idx_col), pl.col("a_toks").list.unique().alias("k")).explode("k")
    r = r.filter(
        (pl.col("k").str.len_chars() >= 4)
        & ~pl.col("k").str.contains(r"\d")
        & ~pl.col("k").is_in(addr_stop)
    ).with_columns(pl.lit("R").alias("ktype"))
    parts.append(r.select(idx_col, "ktype", "k"))

    b = df.select(
        pl.col(idx_col),
        pl.col("a_toks").list.slice(0, 30).list.eval(
            pl.concat_str([pl.element(), pl.element().shift(-1)], separator=" ")
        ).list.drop_nulls().list.unique().alias("k"),
    ).explode("k").drop_nulls()
    parts.append(b.with_columns(pl.lit("B").alias("ktype")).select(idx_col, "ktype", "k"))

    # Unordered pairs of the first 5 informative name tokens.
    nt = df.select(pl.col(idx_col), pl.col("n_toks").list.unique().list.sort().list.slice(0, 5).alias("x"))
    nt = nt.filter(pl.col("x").list.len() >= 2).explode("x").rename({"x": "a"})
    nb = nt.join(nt.rename({"a": "b"}), on=idx_col).filter(pl.col("a") < pl.col("b"))
    parts.append(nb.select(
        pl.col(idx_col), pl.lit("NB").alias("ktype"),
        pl.concat_str(["a", "b"], separator=" ").alias("k"),
    ))

    an = df.filter(pl.col("a_num") != "").select(
        pl.col(idx_col), pl.col("a_num"), pl.col("n_toks").list.unique().alias("tok")
    ).explode("tok").filter(pl.col("tok").str.len_chars() >= 3)
    parts.append(an.select(
        pl.col(idx_col), pl.lit("AN").alias("ktype"),
        pl.concat_str(["a_num", "tok"], separator="|").alias("k"),
    ))

    keys = pl.concat(parts)
    return keys.select(
        pl.col(idx_col).cast(pl.UInt32),
        pl.col("ktype").cast(pl.Utf8),
        pl.concat_str(["ktype", "k"], separator="#").hash(seed=7).alias("h"),
    )


def _index_targets(t_keys: pl.DataFrame) -> pl.DataFrame:
    """Attach df and per-key weight to target keys; drop keys above their cap."""
    df_t = t_keys.group_by("h").agg(pl.len().alias("df"), pl.first("ktype"))
    caps = pl.DataFrame({
        "ktype": list(KEY_CAPS), "cap": list(KEY_CAPS.values()),
        "wt": list(KEY_WEIGHTS.values()), "bit": [KEY_BITS[k] for k in KEY_CAPS],
    })
    df_t = df_t.join(caps, on="ktype").filter(pl.col("df") <= pl.col("cap"))
    df_t = df_t.with_columns(
        (pl.col("wt") / (pl.col("df").cast(pl.Float32) + 1.0).log(2)).cast(pl.Float32).alias("w")
    )
    return t_keys.select("t", "h").join(df_t.select("h", "w", pl.col("bit").cast(pl.UInt8)), on="h")


def pair_similarity(
    q: pl.DataFrame, t: pl.DataFrame, pairs: pl.DataFrame, sub_chunk: int = 2_000_000
) -> pl.DataFrame:
    """Add rapidfuzz similarities of name (sim_n) and address (sim_a), in [0, 1].

    sim_n is the best of: token-set ratio on the core name, on the full name
    (alias records), and plain ratio on the no-space names (domains/handles).
    Also carries t_native / t_aempty flags of the S2/S3 record.
    """
    out = []
    for s in range(0, pairs.height, sub_chunk):
        p = pairs.slice(s, sub_chunk)
        qs = p.select("q").join(
            q.select("q", "n_core", "n_nospace", "a_norm"), on="q", how="left", maintain_order="left")
        ts = p.select("t").join(
            t.select("t", "n_core", "n_full", "n_nospace", "a_norm", "is_native", "a_empty"),
            on="t", how="left", maintain_order="left")
        qn = qs["n_core"].to_list()
        sim_n = process.cpdist(qn, ts["n_core"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        sim_nf = process.cpdist(qn, ts["n_full"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        sim_ns = process.cpdist(qs["n_nospace"].to_list(), ts["n_nospace"].to_list(),
                                scorer=fuzz.ratio, workers=-1)
        sim_a = process.cpdist(qs["a_norm"].to_list(), ts["a_norm"].to_list(),
                               scorer=fuzz.token_set_ratio, workers=-1)
        best_n = sim_n.clip(min=sim_nf).clip(min=sim_ns)
        out.append(p.with_columns(
            sim_n=pl.Series((best_n / 100.0).astype("float32")),
            sim_a=pl.Series((sim_a / 100.0).astype("float32")),
            t_native=ts["is_native"],
            t_aempty=ts["a_empty"],
        ))
    return pl.concat(out)


def rerank_score() -> pl.Expr:
    """Stage-2 ranking score: name + address similarity plus a small key bonus.

    A missing signal is treated as neutral (0.7) rather than as a mismatch:
    native-script names cannot be compared to Latin S1 names, and empty
    addresses carry no evidence either way.
    """
    name_term = pl.when(pl.col("t_native")).then(0.7).otherwise(pl.col("sim_n"))
    addr_term = pl.when(pl.col("t_aempty")).then(0.7).otherwise(pl.col("sim_a"))
    return (name_term + addr_term + 0.05 * pl.col("bscore").clip(upper_bound=10.0)).alias("rscore")


def generate_candidates(
    q: pl.DataFrame,
    t: pl.DataFrame,
    top_k: int = 30,
    pool_k: int = 100,
    top_rev: int = 3,
    chunk: int = 100_000,
    verbose: bool = True,
) -> pl.DataFrame:
    """Return candidate pairs (q, t, bscore, kbits, sim_n, sim_a, rscore, brank).

    ``q`` / ``t`` are normalised frames with integer row indices in columns
    ``q`` and ``t``. ``brank`` is the pair's stage-2 rank among its S1's
    candidates (1 = best).
    """
    t_index = _index_targets(record_keys(t, "t"))
    q_keys = record_keys(q, "q").select("q", "h")
    fwd = []
    rev = None  # running global top-`top_rev` S1 per S2/S3 record (bounded: <= top_rev * |t| rows)
    q_max = int(q["q"].max()) + 1
    for start in range(0, q_max, chunk):
        qk = q_keys.filter((pl.col("q") >= start) & (pl.col("q") < start + chunk))
        pairs = (
            qk.join(t_index, on="h")
            .group_by("q", "t")
            .agg(pl.col("w").sum().alias("bscore"), pl.col("bit").unique().sum().alias("kbits"))
            # float sums depend on summation order; round so ranks are reproducible
            .with_columns(pl.col("bscore").round(4))
        )
        pool = ranked(pairs, "bscore", "q", "t", "_r").filter(pl.col("_r") <= pool_k).drop("_r")
        pool = pair_similarity(q, t, pool).with_columns(rerank_score().round(4))
        fwd.append(ranked(pool, "rscore", "q", "t", "_r").filter(pl.col("_r") <= top_k).drop("_r"))
        # Merge this chunk's reverse candidates into the running global top-k per record
        # after EVERY chunk. Accumulating per-chunk lists instead grows with the pool
        # size (pool 600 -> hundreds of millions of rows; US train OOM'd at 30 GB).
        # Top-k is decomposable and the tie-break is deterministic, so the result is identical.
        chunk_rev = ranked(pool, "rscore", "t", "q", "_r").filter(pl.col("_r") <= top_rev).drop("_r")
        rev = chunk_rev if rev is None else pl.concat([rev, chunk_rev])
        rev = ranked(rev, "rscore", "t", "q", "_r").filter(pl.col("_r") <= top_rev).drop("_r")
        if verbose:
            print(f"    chunk {start:,}: stage1 {pairs.height:,} pool {pool.height:,} rev {rev.height:,}", flush=True)
    del t_index, q_keys  # the key index is the largest object; free it before the merge
    fwd = pl.concat(fwd)
    cands = pl.concat([fwd, rev]).unique(subset=["q", "t"])
    return ranked(cands, "rscore", "q", "t", "brank").with_columns(pl.col("brank").cast(pl.UInt16))


def blocking_recall(cands: pl.DataFrame, truth_idx: pl.DataFrame, n_q: int) -> dict:
    """Pair recall, full-cluster recall and candidates/S1 for index-space pairs.

    ``truth_idx`` has columns (q, t) for the true matches of this country.
    """
    hit = truth_idx.join(
        cands.select("q", "t").with_columns(pl.lit(True).alias("found")), on=["q", "t"], how="left"
    ).with_columns(pl.col("found").fill_null(False))
    per_q = hit.group_by("q").agg(pl.col("found").all().alias("all_found"))
    return {
        "pair_recall": round(float(hit["found"].mean()), 4),
        "cluster_recall": round(float(per_q["all_found"].mean()), 4),
        "cands_per_s1": round(cands.height / n_q, 2),
        "n_pairs": cands.height,
    }
