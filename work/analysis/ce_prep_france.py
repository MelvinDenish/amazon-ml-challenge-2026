"""France adaptation set for the XLM-R cross-encoder (France has no train labels).

France is the weakest country by the model's own estimate after the CE stack (expected F0.5 0.984
vs US 0.991 / India 0.994; uncertain pairs per S1 0.28 vs 0.17 / 0.06) and the CEs never saw French
text. We continue XLM-R training on a mix of
  * 300k original train pairs (keeps US/India behaviour; eval S1s never included),
  * France test pairs with confident labels from the current best stack: positives p >= 0.99,
    hard negatives p <= 0.01 among each S1's 5 best-ranked candidates,
  * synthetic French sibling negatives: a confident positive record + a French qualifier
    (Développement, Holding, Participations, Distribution, Groupe, International, France, SARL, SAS)
    at a nearby house number,
so the CE sees real French noise (R./Rue, N°, accents, départements/régions, SAS/SARL) and the
French distractor pattern. Writes kaggle_jobs/bedrock_ce/{ce_train_fr.parquet, syn_records_fr.parquet}.
"""
import re
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402

A = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts"
OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
WORDS = ["Développement", "Holding", "Participations", "Distribution", "Groupe", "International", "France", "SARL", "SAS"]
rng = np.random.default_rng(3)


def main() -> None:
    t = pl.scan_parquet(f"{A}/oof/test_blend_v7_ce_v12x.parquet").filter(pl.col("country") == "France").collect()
    t = t.with_columns(rk=pl.col("p").rank("ordinal", descending=True).over("s1_id"))
    pos = t.filter(pl.col("p") >= 0.99).sample(150_000, seed=1)
    neg = t.filter((pl.col("p") <= 0.01) & (pl.col("rk") <= 5)).sample(150_000, seed=2)
    rec = pl.concat([load_records("test", s, "France") for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), "business_name", "business_address")
    src = pos.sample(60_000, seed=3).join(rec, on="cand_id").filter(pl.col("business_address").str.contains(r"\d"))
    rows, pairs = [], []
    for i, (s1, name, addr) in enumerate(src.select("s1_id", "business_name", "business_address").iter_rows()):
        m = re.search(r"\d+", addr)
        off = int(rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50]) * rng.choice([1, 1, -1]))
        w = str(rng.choice(WORDS))
        w = w.upper() if name.isupper() else (w.lower() if name.islower() else w)
        rows.append((f"SYNFR-{i}", f"{name} {w}", addr[:m.start()] + str(max(1, int(m.group()) + off)) + addr[m.end():]))
        pairs.append((s1, f"SYNFR-{i}"))
    syn = pl.DataFrame(pairs, schema=["s1_id", "cand_id"], orient="row").with_columns(pl.lit(0, dtype=pl.Int32).alias("y"))
    orig = pl.read_parquet(f"{OUT}/ce_train2.parquet").sample(300_000, seed=4).select("s1_id", "cand_id", pl.col("y").cast(pl.Int32))
    fr = pl.concat([pos.select("s1_id", "cand_id", pl.lit(1, dtype=pl.Int32).alias("y")),
                    neg.select("s1_id", "cand_id", pl.lit(0, dtype=pl.Int32).alias("y")), syn])
    out = pl.concat([orig, fr])
    out.write_parquet(f"{OUT}/ce_train_fr.parquet")
    pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address"], orient="row").write_parquet(
        f"{OUT}/syn_records_fr.parquet")
    print("France adaptation pairs:", out.height, "| France pos", pos.height, "hard neg", neg.height,
          "synthetic sib", syn.height, "| orig", orig.height)


if __name__ == "__main__":
    main()
