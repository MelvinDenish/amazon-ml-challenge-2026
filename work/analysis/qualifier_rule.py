"""Added-qualifier veto: names that ADD a word which (on train) never marks the same business.

Found in the test data: branch/sibling distractors "<same name> Holdings / Group / Riverside /
Downtown / Industries / Exports ..." at nearby numbers are 3-15x more frequent on test than on
train, and on train pairs adding those words match ~0% of the time. The GBM cannot see WHICH word
was added, so it gives them p ~0.15-0.2 and accepts some.

Honest validation (cross-fitted by S1 halves): the zero-rate word list is learned on half A and the
rule is evaluated with the full pipeline on half B, and vice versa. Then the list is learned on all
train pairs and applied to the test predictions (label-free change counts per country).

    python qualifier_rule.py <oof.parquet> <test.parquet> <out_test.parquet>
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from train import evaluate  # noqa: E402

MIN_N, MAX_RATE, FACTOR = 30, 0.01, 0.02
TOK = (pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\p{L}\p{N} ]", " ").str.split(" ")
       .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique())


def with_added(pairs: pl.DataFrame, split: str, country: str) -> pl.DataFrame:
    """Attach the list of name words present only on the candidate side (plausible pairs only)."""
    q = load_records(split, "source1", country).select(pl.col("entity_id").alias("s1_id"), TOK.alias("qt"))
    t = pl.concat([load_records(split, s, country) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"))
    d = pairs.join(q, on="s1_id").join(t, on="cand_id")
    return d.with_columns(pl.col("tt").list.set_difference(pl.col("qt")).alias("add")).drop("qt", "tt")


def zero_words(d: pl.DataFrame) -> set[str]:
    """Words whose addition matched <= MAX_RATE of the time over >= MIN_N plausible train pairs."""
    s = d.select("add", "y").explode("add").drop_nulls().group_by("add").agg(pl.len().alias("n"), pl.col("y").mean().alias("r"))
    return set(s.filter((pl.col("n") >= MIN_N) & (pl.col("r") <= MAX_RATE))["add"].to_list())


def veto(pred: pl.DataFrame, flagged: pl.DataFrame) -> pl.DataFrame:
    """Down-weight flagged pairs (odds x FACTOR)."""
    f = flagged.select("s1_id", "cand_id").with_columns(pl.lit(True).alias("_f"))
    o = pl.col("p") / (1 - pl.col("p").clip(0, 1 - 1e-6))
    adj = (o * FACTOR) / (1 + o * FACTOR)
    return (pred.join(f, on=["s1_id", "cand_id"], how="left")
            .with_columns(pl.when(pl.col("_f")).then(adj).otherwise(pl.col("p")).cast(pl.Float32).alias("p")).drop("_f"))


def flag(d: pl.DataFrame, words: set[str]) -> pl.DataFrame:
    return d.filter(pl.col("add").list.eval(pl.element().is_in(list(words))).list.any())


def main() -> None:
    oof_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    oof = pl.read_parquet(oof_path)
    half = (pl.col("s1_id").hash(seed=11) % 2).alias("half")
    plaus = []
    for c in ("India", "US"):
        o = oof.filter((pl.col("country") == c) & (pl.col("p") > 0.02)).select("s1_id", "cand_id", "y", "p")
        plaus.append(with_added(o, "train", c).with_columns(pl.lit(c).alias("country")))
    plaus = pl.concat(plaus).with_columns(half)
    oofh = oof.with_columns(half)
    for h in (0, 1):
        words = zero_words(plaus.filter(pl.col("half") != h))
        ev = oofh.filter(pl.col("half") == h).drop("half")
        fl = flag(plaus.filter(pl.col("half") == h), words)
        print(f"half {h}: {len(words)} zero-rate words learned on the other half; flagged pairs {fl.height:,} "
              f"(true matches among them {int(fl['y'].sum())})", flush=True)
        for c in ("India", "US"):
            e = ev.filter(pl.col("country") == c)
            for tag, x in (("base", e), ("veto", veto(e, fl))):
                r = evaluate(exclusive_owner_prob(x, "p"), [c])["ef05"][c]
                print(f"   {c:6s} {tag:5s} (owner prob) {r:.5f}", flush=True)
    words = zero_words(plaus)
    print(f"\nALL-TRAIN list: {len(words)} words, e.g.", sorted(words)[:60], flush=True)
    parts = []
    for c in ("France", "India", "US"):
        t = pl.scan_parquet(test_path).filter(pl.col("country") == c).collect()
        if "p" not in t.columns:
            t = t.rename({[x for x in t.columns if x not in ("s1_id", "cand_id", "country")][0]: "p"})
        fl = flag(with_added(t.filter(pl.col("p") > 0.02).select("s1_id", "cand_id", "p"), "test", c), words)
        tv = veto(t, fl)
        n = t["s1_id"].n_unique()
        b = select_expected_f05(one_owner(exclusive_owner_prob(t, "p"), "p"), "p").select("s1_id", "cand_id")
        a = select_expected_f05(one_owner(exclusive_owner_prob(tv, "p"), "p"), "p").select("s1_id", "cand_id")
        removed = b.join(a, on=["s1_id", "cand_id"], how="anti").height
        added_ = a.join(b, on=["s1_id", "cand_id"], how="anti").height
        print(f"TEST {c:6s}: flagged {fl.height / n * 1000:.1f}/1k S1, selections removed {removed / n * 1000:.1f}/1k, "
              f"added {added_ / n * 1000:.1f}/1k, matches/S1 {b.height / n:.3f} -> {a.height / n:.3f}", flush=True)
        parts.append(tv.select("s1_id", "cand_id", "country", "p"))
    pl.concat(parts).write_parquet(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
