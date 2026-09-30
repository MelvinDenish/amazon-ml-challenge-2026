"""CE v3 training set = CE v2 pairs + synthetic test-like sibling negatives.

Test is full of '<name> + qualifier / legal form' records at a nearby house number (US 31 -> 112
per 1k S1 in 'word|near|anch'); train has few, so the CE sees few. For S1s of the CE-train pool
(never an eval S1) we take one TRUE record, add a qualifier drawn from the test vocabulary of that
country (suffix/inside/prefix, matching the record's casing) and move its house number by an
offset drawn from the test offset distribution -> label 0. Real noisy true pairs (incl. words added
at the SAME number, e.g. 'Center', legal forms) stay in the set as positives, so the CE cannot
learn 'added word => different business'; it must learn 'added qualifier AND other number'.
Writes kaggle_jobs/bedrock_ce/{ce_train3.parquet, syn_records.parquet}.
"""
import re
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records  # noqa: E402

OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
LIB = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\sibling_deltas.parquet"
N = {"US": 160_000, "India": 100_000}
WORDS = {
    "US": (["Holdings", "Group", "Partners"] * 4 + ["Riverside", "Downtown", "Lakeside", "Metro", "Uptown", "Midtown",
           "Central", "Greater", "North", "South", "East", "West", "Northside", "Southside", "Eastgate", "Westgate",
           "Coastal", "Harbor", "Summit", "Valley", "Highland"] + ["Inc", "LLC", "Corp", "Co", "Ltd", "LP"] * 2),
    "India": (["Industries", "Enterprises", "Infratech", "Overseas", "Exports", "Ventures", "Group", "Holdings"] * 2
              + ["Private Limited", "Pvt Ltd", "Limited", "LLP", "Public Limited"]),
}
rng = np.random.default_rng(7)


def styled(word: str, template: str) -> str:
    """Match the template name's casing."""
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word


def insert(name: str, word: str) -> str:
    """Put the qualifier at the end (65%), before the last token (25%) or in front (10%)."""
    toks = name.split()
    r = rng.random()
    if r < 0.65 or len(toks) < 2:
        return f"{name} {word}"
    if r < 0.90:
        return " ".join(toks[:-1] + [word, toks[-1]])
    return f"{word} {name}"


def main() -> None:
    lib = pl.read_parquet(LIB)
    base = pl.read_parquet(f"{OUT}/ce_train2.parquet")
    pool_s1 = base.select("s1_id").unique()
    gt = load_ground_truth_pairs().join(pool_s1, on="s1_id", how="semi")
    syn_rows, syn_pairs = [], []
    k = 0
    for c, n in N.items():
        off = lib.filter(pl.col("country") == c)["offset"].to_numpy()
        near = off[np.abs(off) <= 60]
        t = pl.concat([load_records("train", s, c) for s in ("source2", "source3")]).select(
            pl.col("entity_id").alias("cand_id"), "business_name", "business_address")
        g = (gt.join(t, on="cand_id").filter(pl.col("business_address").str.contains(r"\d"))
             .unique("s1_id").sample(n, seed=11))
        for s1, name, addr in g.select("s1_id", "business_name", "business_address").iter_rows():
            o = int(rng.choice(near) if rng.random() < 0.7 else rng.choice(off))
            o = o if o != 0 else 2
            m = re.search(r"\d+", addr)
            new_addr = addr[:m.start()] + str(max(1, int(m.group()) + o)) + addr[m.end():]
            word = styled(str(rng.choice(WORDS[c])), name)
            eid = f"SYN-{k}"
            k += 1
            syn_rows.append((eid, insert(name, word), new_addr))
            syn_pairs.append((s1, eid, c))
        print(c, "synthetic siblings", n, flush=True)
    recs = pl.DataFrame(syn_rows, schema=["entity_id", "business_name", "business_address"], orient="row")
    pairs = pl.DataFrame(syn_pairs, schema=["s1_id", "cand_id", "country"], orient="row").with_columns(
        pl.lit(0, dtype=base["y"].dtype).alias("y"), pl.lit(0.5, dtype=base["p"].dtype).alias("p"))
    recs.write_parquet(f"{OUT}/syn_records.parquet")
    out = pl.concat([base, pairs.select(base.columns)])
    out.write_parquet(f"{OUT}/ce_train3.parquet")
    print("ce_train3 pairs", out.height, "pos rate", round(float(out["y"].mean()), 3))
    for r in recs.head(8).to_dicts():
        print(r)


if __name__ == "__main__":
    main()
