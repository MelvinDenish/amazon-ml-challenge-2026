"""CE veto for CONFIDENT pairs in risky groups (twins / legal-form conflict / empty address).

The GBM routed only 0.005<p<0.995 to the cross-encoders; confident pairs in the groups where test
has more distractors were never checked. All three CEs now scored them. Rule: veto (odds x 1e-3)
when the MAX CE logit over the three models is below T (all three reject).
  1. held-out eval S1s: how many true matches would each T veto (must be ~0),
  2. test: vetoes per 1k S1 per group vs the excess of test selections over train,
  3. writes the vetoed test prediction for the chosen T.

    python ce_veto.py <T> <base_test_pred> <out>
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from labelshift import group_expr  # noqa: E402

A = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts"
X = f"{A}/k9e_out"
CE_LIST = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
TAGS = ("v1", "v2", "xlmr")


def scores(split: str) -> pl.DataFrame:
    """The three CE logits per pair plus their max and mean."""
    d = None
    for t in TAGS:
        s = pl.read_parquet(f"{X}/{split}_x_scores_{t}.parquet").rename({"ce": t})
        d = s if d is None else d.join(s, on=["s1_id", "cand_id"])
    return d.with_columns(ce_max=pl.max_horizontal(TAGS), ce_mean=pl.mean_horizontal(TAGS))


def main() -> None:
    T, base, out = float(sys.argv[1]), sys.argv[2], sys.argv[3]
    ev = pl.read_parquet(f"{CE_LIST}/ce_eval_x.parquet").join(scores("eval"), on=["s1_id", "cand_id"])
    n_ev = (pl.scan_parquet(f"{A}/oof/blend_v7.parquet").filter((pl.col("s1_id").hash(seed=45) % 10_000) < 5000)
            .select(pl.col("s1_id").n_unique()).collect().item())
    print(f"held-out: {ev.height:,} confident risky pairs ({(1 - ev['y'].mean()) * ev.height:.0f} non-matches), {n_ev:,} S1")
    for t in (-1.0, -2.0, -3.0, -4.0, -5.0):
        v = ev.filter(pl.col("ce_max") < t)
        print(f"   T={t:5.1f}: vetoed {v.height:5d} ({v.height / n_ev * 1000:.2f}/1k S1)  of which TRUE {int(v['y'].sum())}")
    te = pl.read_parquet(f"{CE_LIST}/ce_test_x.parquet").join(scores("test"), on=["s1_id", "cand_id"])
    g = pl.concat([pl.read_parquet(f"{A}/groups/test_{c}.parquet", columns=["s1_id", "cand_id", "twin_flag", "num_absdiff",
                   "form_disjoint", "t_a_empty"]) for c in ("France", "India", "US")]).with_columns(group_expr()).select(
        "s1_id", "cand_id", "group")
    te = te.join(g, on=["s1_id", "cand_id"], how="left")
    n_te = pl.scan_parquet(base).group_by("country").agg(pl.col("s1_id").n_unique().alias("n")).collect()
    for t in (-1.0, -2.0, -3.0, -4.0, -5.0):
        v = te.filter(pl.col("ce_max") < t).group_by("country", "group").len().join(n_te, on="country")
        v = v.with_columns((pl.col("len") / pl.col("n") * 1000).round(1).alias("per_1k")).select(
            "country", "group", "per_1k").sort("country", "group")
        print(f"   TEST T={t:5.1f}:", v.to_dicts())
    veto = te.filter(pl.col("ce_max") < T).select("s1_id", "cand_id").with_columns(pl.lit(True).alias("_v"))
    b = pl.read_parquet(base)
    o = pl.col("p").cast(pl.Float64).clip(1e-7, 1 - 1e-7)
    o = o / (1 - o) * 1e-3
    b = b.join(veto, on=["s1_id", "cand_id"], how="left").with_columns(
        pl.when(pl.col("_v")).then(o / (1 + o)).otherwise(pl.col("p")).cast(pl.Float32).alias("p")).drop("_v")
    b.write_parquet(out)
    print(f"vetoed {veto.height:,} test pairs at T={T}; wrote {out}")


if __name__ == "__main__":
    main()
