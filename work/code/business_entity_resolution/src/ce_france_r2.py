"""Round-2 France adaptation set (after the leaderboard confirmed the France rules).

Pseudo-labels come from the final France predictions of the validated recipe (france_final.py output):
  * positives: p >= 0.99, plus the generator's filler variants with p >= 0.9 (swap_rules.filler_swap_pairs),
  * negatives: p <= 0.01 among each S1's 5 best candidates, plus co-located vocabulary-noun swaps
    (swap_rules.vocab_swap_pairs, extended) taken from the full blocked set,
  * synthetic sibling negatives: a confident positive record + a PURE sibling qualifier or legal form
    (Holding, Participations, Distribution, International, SARL, SAS, EURL, SCI, SASU) at a nearby number.
    Round 1 also used Développement / Groupe / France, which are filler words as well and taught the
    encoders to reject true filler variants.
Mixed with 600k original train pairs. Writes ce/train_r2.parquet (texts attached; the input format of
modal_ce.py) and ce/test_fr.parquet (France test pairs to score).

    python ce_france_r2.py --pred test_final --full test_final_nocascade
"""
import argparse
import re

import numpy as np
import polars as pl

from config import ARTIFACT_DIR
from cross_encoder import attach, texts
from io_utils import load_records
from swap_rules import filler_swap_pairs, vocab_swap_pairs

A = str(ARTIFACT_DIR)
CE = ARTIFACT_DIR / "ce"
WORDS = ["Holding", "Participations", "Distribution", "International", "SARL", "SAS", "EURL", "SCI", "SASU"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="test_final", help="final predictions (cascade candidate set)")
    ap.add_argument("--full", default="test_final_nocascade", help="final predictions on the full blocked set")
    ap.add_argument("--orig", default="train_full.parquet", help="original train pairs with texts (modal_ce input)")
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    t = pl.scan_parquet(f"{A}/oof/{a.pred}.parquet").filter(pl.col("country") == "France").select("s1_id", "cand_id", "p").collect()
    t = t.with_columns(rk=pl.col("p").rank("ordinal", descending=True).over("s1_id"))
    pos = t.filter(pl.col("p") >= 0.99).sample(160_000, seed=1).select("s1_id","cand_id", y=pl.lit(1))
    fill = filler_swap_pairs(t, "test", "France").join(t, on=["s1_id","cand_id"]).filter(pl.col("p") >= 0.9).select("s1_id","cand_id", y=pl.lit(1))
    # the full blocked set (no cascade) still carries the noun swaps the cascade dropped
    full = pl.scan_parquet(f"{A}/oof/{a.full}.parquet").filter(pl.col("country")=="France").select("s1_id","cand_id","p").collect()
    noun = vocab_swap_pairs(full, "test", "France", extended=True).sample(fraction=1.0, seed=2).head(60_000).with_columns(y=pl.lit(0))
    neg = t.filter((pl.col("p") <= 0.01) & (pl.col("rk") <= 5)).sample(110_000, seed=3).select("s1_id","cand_id", y=pl.lit(0))
    print("pos", pos.height, "filler pos", fill.height, "noun neg", noun.height, "hard neg", neg.height, flush=True)
    # synthetic siblings: confident positive record + PURE sibling qualifier / legal form at a nearby number (never a filler word)

    rec = pl.concat([load_records("test", s, "France") for s in ("source2","source3")]).select(pl.col("entity_id").alias("cand_id"), "business_name", "business_address")
    src = pos.sample(45_000, seed=4).join(rec, on="cand_id").filter(pl.col("business_address").str.contains(r"\d"))
    rows, pairs = [], []
    for i, (s1, name, addr) in enumerate(src.select("s1_id","business_name","business_address").iter_rows()):
        m = re.search(r"\d+", addr)
        off = int(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]) * rng.choice([1, -1]))
        w = str(rng.choice(WORDS)); w = w.upper() if name.isupper() else w
        rows.append((f"SYNFR2-{i}", f"{name} {w}", addr[:m.start()] + str(max(1, int(m.group()) + off)) + addr[m.end():]))
        pairs.append((s1, f"SYNFR2-{i}"))
    syn = pl.DataFrame(pairs, schema=["s1_id","cand_id"], orient="row").with_columns(y=pl.lit(0))
    fr = pl.concat([pos, fill, noun, neg]).unique(["s1_id","cand_id"], keep="first")
    tx = texts("test")
    tx_syn = pl.DataFrame(rows, schema=["entity_id","business_name","business_address"], orient="row").select(
        "entity_id", pl.concat_str([pl.col("business_name"), pl.col("business_address")], separator=" ; ").str.slice(0, 400).alias("txt"))
    fr = attach(fr, tx)
    syn = attach(syn, pl.concat([tx.filter(pl.col("entity_id").is_in(syn["s1_id"].implode())), tx_syn]))
    orig = pl.read_parquet(CE / a.orig).sample(600_000, seed=5)
    out = pl.concat([orig.select("s1_id","cand_id",pl.col("y").cast(pl.Int32),"a_txt","b_txt"),
                     fr.select("s1_id","cand_id",pl.col("y").cast(pl.Int32),"a_txt","b_txt"),
                     syn.select("s1_id","cand_id",pl.col("y").cast(pl.Int32),"a_txt","b_txt")]).sample(fraction=1.0, seed=6, shuffle=True)
    print("train_r2:", out.height, "positives", out["y"].mean(), "| France", fr.height, "syn", syn.height, "empty texts", (out["a_txt"]=="").sum(), (out["b_txt"]=="").sum(), flush=True)
    out.write_parquet(CE / "train_r2.parquet")
    te = attach(pl.read_parquet(CE / "ce_test.parquet"), tx)
    frs1 = load_records("test", "source1", "France").select(pl.col("entity_id").alias("s1_id"))
    tf = te.join(frs1, on="s1_id", how="semi"); tf.write_parquet(CE / "test_fr.parquet")
    print("test_fr pairs", tf.height)


if __name__ == "__main__":
    main()
