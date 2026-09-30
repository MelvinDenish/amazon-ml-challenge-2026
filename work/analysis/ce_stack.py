"""Stack the cross-encoder onto a GBM and measure the full pipeline honestly (review item C).

    python ce_stack.py <base_oof.parquet> <ce_eval_scores.parquet> [--test <base_test.parquet> <ce_test_scores.parquet> <out.parquet>]

Evaluation universe: every S1 in (v07-blend OOF S1s) & (v08 sample) & (base OOF), with ALL its
pairs, so the score is the real macro F0.5 over that population, not over hard cases only.
Routing: a pair is re-scored iff it has a CE score (the v07 uncertainty band), identical on
validation and test. Stacker: logistic regression on [logit p, ce logit, product], cross-fitted
2-fold by S1 hash inside the evaluation S1s. Non-routed pairs keep p. Reported with and without
exclusive-owner probabilities. With --test, the stacker is refit on all evaluation pairs and
applied to the routed test pairs; the result is written as a test prediction parquet.
"""
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from postprocess import exclusive_owner_prob  # noqa: E402
from train import evaluate  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
SEED = 42


def X(df: pl.DataFrame) -> np.ndarray:
    """Stacker inputs: logit of the GBM probability, CE logit, and their product."""
    p = np.clip(df["p"].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    ce = df["ce"].to_numpy().astype(np.float64)
    return np.column_stack([lp, ce, lp * ce])


def main() -> None:
    base_path, ce_path = sys.argv[1], sys.argv[2]
    v07_s1 = pl.scan_parquet(f"{K5}/oof/blend_v7.parquet").select("s1_id").unique().collect()
    base = (pl.read_parquet(base_path).join(v07_s1, on="s1_id", how="semi")
            .filter((pl.col("s1_id").hash(seed=SEED + 3) % 10_000) < 5000))
    ce = pl.read_parquet(ce_path)
    d = base.join(ce, on=["s1_id", "cand_id"], how="left")
    routed = d.filter(pl.col("ce").is_not_null())
    print(f"eval S1s {d['s1_id'].n_unique():,} pairs {d.height:,} routed {routed.height:,} "
          f"(CE pairs matched {routed.height:,}/{ce.height:,})", flush=True)
    fold = (routed["s1_id"].hash(seed=99) % 2).to_numpy()
    newp = np.zeros(routed.height)
    for k in (0, 1):
        m = LogisticRegression(C=1.0, max_iter=500).fit(X(routed)[fold != k], routed["y"].to_numpy()[fold != k])
        newp[fold == k] = m.predict_proba(X(routed)[fold == k])[:, 1]
    full = LogisticRegression(C=1.0, max_iter=500).fit(X(routed), routed["y"].to_numpy())
    print("stacker coef [logit p, ce, product]:", np.round(full.coef_[0], 3), "intercept", round(float(full.intercept_[0]), 3))
    stacked = d.join(routed.select("s1_id", "cand_id").with_columns(pl.Series("p_st", newp.astype(np.float32))),
                     on=["s1_id", "cand_id"], how="left").with_columns(pl.coalesce("p_st", "p").alias("p")).drop("p_st", "ce")
    plain = d.drop("ce")
    if "--save-oof" in sys.argv:
        out_oof = sys.argv[sys.argv.index("--save-oof") + 1]
        stacked.write_parquet(out_oof)
        print("wrote stacked OOF", out_oof, flush=True)
    for tag, df in (("base", plain), ("base+CE", stacked)):
        for own in (False, True):
            parts = {}
            for c in ("India", "US"):
                x = df.filter(pl.col("country") == c)
                parts[c] = evaluate(exclusive_owner_prob(x, "p") if own else x, [c])["ef05"][c]
            n = {c: d.filter(pl.col("country") == c)["s1_id"].n_unique() for c in parts}
            overall = sum(parts[c] * n[c] for c in parts) / sum(n.values())
            print(f"  {tag:8s} owner_prob={own!s:5s} India {parts['India']:.5f} US {parts['US']:.5f} overall {overall:.5f}", flush=True)
    if "--test" in sys.argv:
        i = sys.argv.index("--test")
        t_base, t_ce, out = sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3]
        tce = pl.read_parquet(t_ce)
        parts = []
        for c in ("France", "India", "US"):
            t = pl.scan_parquet(t_base).filter(pl.col("country") == c).collect()
            name = [x for x in t.columns if x not in ("s1_id", "cand_id", "country", "p")]
            if "p" not in t.columns:
                t = t.rename({name[0]: "p"})
            r = t.join(tce, on=["s1_id", "cand_id"], how="inner")
            pr = full.predict_proba(X(r))[:, 1].astype(np.float32)
            t = t.join(r.select("s1_id", "cand_id").with_columns(pl.Series("p_st", pr)), on=["s1_id", "cand_id"], how="left")
            print(f"  test {c}: pairs {t.height:,} routed {r.height:,}  CE logit mean {r['ce'].mean():.3f} "
                  f"sd {r['ce'].std():.3f}  mean p {r['p'].mean():.4f} -> stacked {pr.mean():.4f}", flush=True)
            parts.append(t.with_columns(pl.coalesce("p_st", "p").alias("p")).select("s1_id", "cand_id", "country", "p"))
        pl.concat(parts).write_parquet(out)
        print("wrote", out)


if __name__ == "__main__":
    main()
