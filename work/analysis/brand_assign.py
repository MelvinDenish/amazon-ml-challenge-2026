"""Invented-brand records: match by ADDRESS (they belong to some S1 97-98% of the time).

Records whose whole name is one token that never occurs in any S1 name ('Iriarc', 'Dovaarc',
'Veonoviveo', '@dentcoffee' ...) are aliases or handles: on train 97.4% (US) / 98.2% (India) of them
are a true match of some S1, versus 74% of all records, yet about 12 per 1k S1 are still missed.
For each such record left unassigned by the current pipeline, find S1s sharing its house number and
>=1 address word, score the address similarity, and attach it to the best S1 when the score is high
and clearly ahead of the runner-up. Held-out evaluation on the eval S1s (F0.5 before / after).

    python brand_assign.py <evaloof>
"""
import sys

import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from score import per_entity_f05  # noqa: E402

TOK = (pl.col("business_name").str.to_lowercase().str.replace_all(r"[^a-z ]", " ").str.split(" ")
       .list.eval(pl.element().filter(pl.element() != "")))
LOW = pl.col("business_address").fill_null("").str.to_lowercase()
NUM = LOW.str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1")
WORDS = LOW.str.replace_all(r"[^a-z ]", " ").str.split(" ").list.eval(
    pl.element().filter(pl.element().str.len_chars() >= 4)).list.unique()
CAP = 40


def brand_records(split: str, c: str, vocab) -> pl.DataFrame:
    """Records whose name is a single token never seen in any S1 name."""
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).with_columns(tk=TOK)
    return t.filter((pl.col("tk").list.len() == 1) & pl.col("tk").list.eval(pl.element().is_in(vocab)).list.any().not_()
                    & (pl.col("tk").list.first().str.len_chars() >= 5)).select(
        pl.col("entity_id").alias("cand_id"), "business_address")


def propose(s1: pl.DataFrame, br: pl.DataFrame) -> pl.DataFrame:
    """Best S1 per brand record by address: (cand_id, s1_id, score, margin)."""
    def keys(df, idc):
        return (df.select(pl.col(idc), NUM.alias("n"), WORDS.alias("w")).drop_nulls("n").explode("w").drop_nulls("w")
                .select(idc, pl.concat_str(["n", "w"], separator="|").alias("k")).unique())
    sk, bk = keys(s1, "s1_id"), keys(br, "cand_id")
    df = sk.group_by("k").len("df").filter(pl.col("df") <= CAP)
    pairs = bk.join(sk.join(df, on="k"), on="k").select("cand_id", "s1_id").unique()
    pairs = (pairs.join(br.select("cand_id", pl.col("business_address").alias("ba")), on="cand_id")
             .join(s1.select("s1_id", pl.col("business_address").alias("sa")), on="s1_id"))
    sc = process.cpdist([x.lower() for x in pairs["ba"].fill_null("")], [x.lower() for x in pairs["sa"].fill_null("")],
                        scorer=fuzz.token_set_ratio, workers=-1)
    pairs = pairs.with_columns(score=pl.Series(sc)).sort(["cand_id", "score"], descending=[False, True])
    top = pairs.group_by("cand_id", maintain_order=True).agg(
        pl.col("s1_id").first(), pl.col("score").first(), pl.col("score").slice(1, 1).first().fill_null(0).alias("second"))
    return top.with_columns(margin=pl.col("score") - pl.col("second"))


def main() -> None:
    ev = pl.read_parquet(sys.argv[1])
    gt = load_ground_truth_pairs()
    rules = [(95, 0), (90, 10), (85, 10), (80, 5)]
    for c in ("India", "US"):
        s1_all = load_records("train", "source1", c).select(pl.col("entity_id").alias("s1_id"), "business_name", "business_address")
        vocab = s1_all.select(TOK.alias("t")).explode("t").drop_nulls().unique()["t"].implode()
        o = ev.filter(pl.col("country") == c)
        eval_s1 = o.select("s1_id").unique()
        sel = select_expected_f05(one_owner(exclusive_owner_prob(o, "p"), "p"), "p").select("s1_id", "cand_id")
        br = brand_records("train", c, vocab).join(sel, on="cand_id", how="anti")
        prop = propose(s1_all, br).join(eval_s1, on="s1_id")  # all S1s compete (as on test); keep eval-owned proposals
        truth = gt.join(eval_s1, on="s1_id")
        base = per_entity_f05(truth, sel, eval_s1["s1_id"])["f05"].mean()
        for sc_min, m_min in rules:
            add = prop.filter((pl.col("score") >= sc_min) & (pl.col("margin") >= m_min)).select("s1_id", "cand_id")
            hit = add.join(truth, on=["s1_id", "cand_id"], how="semi").height
            f = per_entity_f05(truth, pl.concat([sel, add]).unique(), eval_s1["s1_id"])["f05"].mean()
            print(f"[{c}] score>={sc_min} margin>={m_min}: add {add.height / eval_s1.height * 1000:.1f}/1k S1, precision "
                  f"{hit / max(add.height, 1):.1%} | F0.5 {base:.5f} -> {f:.5f} ({f - base:+.5f})", flush=True)


if __name__ == "__main__":
    main()
