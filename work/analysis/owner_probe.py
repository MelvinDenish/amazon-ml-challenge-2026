"""Owner-aware assignment probe on the v06 OOF (item 1 of the review follow-up).

Each S2/S3 record belongs to at most one S1 (or to none). Today the pipeline takes
the argmax-p owner and discards the others. Here the scores of all S1s competing for
one record are turned into owner probabilities with an explicit null owner:

    o_s = p_s / (1 - p_s),   q_s = o_s / (1 + sum_s' o_s')     (exclusive owners)

and blended with the raw p (lambda = 0 is the current pipeline). The full pipeline
(one_owner -> expected-F0.5 selection -> macro F0.5) is re-scored per country on the
real labels, and the test-side effect (matches per S1, changed S1s) is reported.
Caveat: the OOF holds a 50% S1 sample, so half of the competing owners are absent;
test competition is denser, so the test-side change is reported alongside.
"""
import sys
import time

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from postprocess import one_owner, select_expected_f05  # noqa: E402
from train import evaluate  # noqa: E402

A = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts"
LAMBDAS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.0, 0.25, 0.5, 1.0]


def owner_prob(df: pl.DataFrame, lam: float) -> pl.DataFrame:
    """Replace p by (1-lam)*p + lam*q, q = exclusive-owner probability with a null owner."""
    if lam == 0:
        return df
    o = pl.col("p").cast(pl.Float64).clip(1e-6, 1 - 1e-6)
    o = o / (1 - o)
    return df.with_columns(
        (((1 - lam) * pl.col("p") + lam * (o / (1 + o.sum().over("cand_id")))).clip(0.0, 1.0)).cast(pl.Float32).alias("p"))


def main() -> None:
    oof = pl.read_parquet(f"{A}/k2_out_v2/artifacts/oof/xgb_v6.parquet")
    for country in ("India", "US"):
        d = oof.filter(pl.col("country") == country)
        n_comp = d.group_by("cand_id").len().filter(pl.col("len") > 1).height
        print(f"\n[{country}] pairs {d.height:,}  cands with >1 S1 {n_comp:,}", flush=True)
        for lam in LAMBDAS:
            t0 = time.time()
            r = evaluate(owner_prob(d, lam), [country])
            print(f"  lambda {lam:4.2f}: ef05 {r['ef05'][country]:.6f}  ({time.time() - t0:.0f}s)", flush=True)
        del d
    del oof
    test = pl.read_parquet(f"{A}/k3_out/artifacts/oof/test_xgb_v6.parquet")
    base = select_expected_f05(one_owner(test, "p"), "p").select("s1_id", "cand_id")
    n_s1 = test["s1_id"].n_unique()
    print(f"\nTEST v06: selected {base.height:,} pairs, {base.height / n_s1:.3f} per S1")
    for lam in LAMBDAS[1:]:
        sel = select_expected_f05(one_owner(owner_prob(test, lam), "p"), "p").select("s1_id", "cand_id")
        changed = pl.concat([base.join(sel, on=["s1_id", "cand_id"], how="anti"),
                             sel.join(base, on=["s1_id", "cand_id"], how="anti")])["s1_id"].n_unique()
        print(f"  lambda {lam:4.2f}: {sel.height / n_s1:.3f} per S1, changed S1s {changed:,} ({changed / n_s1:.2%})")


if __name__ == "__main__":
    main()
