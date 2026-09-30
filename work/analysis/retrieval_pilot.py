"""Retrieval pilot (review item A): WHERE does blocking lose true pairs, and what recovers them?

    python retrieval_pilot.py India 30000

Fixed subset: the first N S1s (sorted id) of one train country, searched against the
FULL S2/S3 pool of that country with the production pool-600 / top-20 settings.
Every true pair is put in exactly one bucket:
  no_key      shares no uncapped blocking key with its S1 (stage 1 never sees it)
  pool_cut    has keys but its stage-1 score rank is > pool_k (600)
  topk_cut    in the pool, re-rank position > top_k (20)    (reverse top-3 not modelled:
              it needs all S1s; the v06 misses file gives the real end-to-end number)
  found       in the forward top-20
Then, for the lost pairs, what WOULD recover them, each measured as
(recovered true pairs, extra candidate pairs added per S1):
  topk 30 / 40, and a character-3-gram name route (TF-IDF-like, top-10 per S1).
"""
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from blocking import _index_targets, pair_similarity, record_keys, rerank_score  # noqa: E402
from build_candidates import BLOCKING_PARAMS  # noqa: E402
from dataset import load_country, truth_in_index_space  # noqa: E402
from postprocess import ranked  # noqa: E402

country, n_sub = sys.argv[1], int(sys.argv[2])
POOL, TOPK = BLOCKING_PARAMS["pool_k"], BLOCKING_PARAMS["top_k"]
t0 = time.time()
q, t = load_country("train", country)
truth_all = truth_in_index_space(q, t)
sub = q.sort("entity_id").head(n_sub).select("q")
q_sub = q.join(sub, on="q", how="semi")
truth = truth_all.join(sub, on="q", how="semi")
print(f"[{country}] q_sub={q_sub.height:,} t={t.height:,} true pairs={truth.height:,} load {time.time() - t0:.0f}s", flush=True)

# ---- production stage 1 + re-rank, keeping ranks for diagnosis
t_index = _index_targets(record_keys(t, "t"))
pairs = (record_keys(q_sub, "q").select("q", "h").join(t_index, on="h")
         .group_by("q", "t").agg(pl.col("w").sum().alias("bscore"), pl.col("bit").unique().sum().alias("kbits"))
         .with_columns(pl.col("bscore").round(4)))
pairs = ranked(pairs, "bscore", "q", "t", "prank")
pool = pair_similarity(q, t, pairs.filter(pl.col("prank") <= POOL)).with_columns(rerank_score().round(4))
pool = ranked(pool, "rscore", "q", "t", "rrank")
print(f"stage1 pairs {pairs.height:,}  pool {pool.height:,}  ({time.time() - t0:.0f}s)", flush=True)

tp = (truth.join(pairs.select("q", "t", "prank"), on=["q", "t"], how="left")
      .join(pool.select("q", "t", "rrank"), on=["q", "t"], how="left"))
tp = tp.with_columns(
    pl.when(pl.col("prank").is_null()).then(pl.lit("no_key"))
    .when(pl.col("prank") > POOL).then(pl.lit("pool_cut"))
    .when(pl.col("rrank") > TOPK).then(pl.lit("topk_cut"))
    .otherwise(pl.lit("found")).alias("bucket"))
print("\nBUCKETS:", tp.group_by("bucket").len().sort("bucket").with_columns(
    (pl.col("len") / tp.height * 100).round(2).alias("pct")).to_dicts())
print("pool_cut true pairs, stage-1 rank quantiles:",
      tp.filter(pl.col("bucket") == "pool_cut")["prank"].quantile(0.5), tp.filter(pl.col("bucket") == "pool_cut")["prank"].quantile(0.9))

# ---- option: larger top-k (only moves topk_cut pairs)
for k in (30, 40):
    rec = tp.filter((pl.col("bucket") == "topk_cut") & (pl.col("rrank") <= k)).height
    extra = pool.filter((pl.col("rrank") > TOPK) & (pl.col("rrank") <= k)).height
    print(f"top_k {k}: recovers {rec:,} true pairs ({rec / truth.height:.2%}), +{extra / q_sub.height:.1f} cands/S1")

# ---- option: character 3-gram route on the no-space name, for lost pairs
lost = tp.filter(pl.col("bucket").is_in(["no_key", "pool_cut"])).select("q", "t")


def grams(df: pl.DataFrame, idx: str) -> pl.DataFrame:
    """Distinct character 3-grams of the no-space name."""
    s = df.select(idx, pl.col("n_nospace").alias("s")).filter(pl.col("s").str.len_chars() >= 3)
    return (s.with_columns(pl.int_ranges(0, pl.col("s").str.len_chars() - 2).alias("i")).explode("i")
            .select(idx, pl.col("s").str.slice(pl.col("i"), 3).alias("g")).unique())


tg = grams(t, "t")
gdf = tg.group_by("g").len("df").filter(pl.col("df") <= 2000)
tg = tg.join(gdf, on="g").with_columns((1.0 / (pl.col("df").cast(pl.Float32) + 1).log(2)).alias("w"))
qg = grams(q_sub, "q")
for top in (5, 10):
    parts = []
    qids = q_sub["q"].to_numpy()
    for s in range(0, len(qids), 5000):
        chunk = qg.filter(pl.col("q").is_in(qids[s:s + 5000].tolist()))
        sc = chunk.join(tg.select("t", "g", "w"), on="g").group_by("q", "t").agg(pl.col("w").sum().alias("gs"))
        sc = sc.with_columns(pl.col("gs").round(4))
        parts.append(ranked(sc, "gs", "q", "t", "_r").filter(pl.col("_r") <= top).select("q", "t"))
    route = pl.concat(parts)
    new = route.join(pool.filter(pl.col("rrank") <= TOPK).select("q", "t"), on=["q", "t"], how="anti")
    rec = lost.join(route, on=["q", "t"], how="semi").height
    print(f"char3 name route top-{top}: recovers {rec:,} of {lost.height:,} lost (no_key+pool_cut) true pairs "
          f"({rec / truth.height:.2%} of all), +{new.height / q_sub.height:.1f} new cands/S1   ({time.time() - t0:.0f}s)", flush=True)

# what do the lost pairs look like?
ex = (lost.join(q.select("q", pl.col("n_core").alias("q_name"), pl.col("a_norm").alias("q_addr")), on="q")
      .join(t.select("t", pl.col("n_core").alias("t_name"), pl.col("a_norm").alias("t_addr"), "a_empty", "is_native"), on="t"))
print("\nlost pairs: t empty address", round(float(ex["a_empty"].mean()), 3), " t native name", round(float(ex["is_native"].mean()), 3))
for r in ex.head(12).to_dicts():
    print("  ", r["q_name"][:40], "|", r["q_addr"][:50], "  <->  ", r["t_name"][:40], "|", (r["t_addr"] or "")[:50])
