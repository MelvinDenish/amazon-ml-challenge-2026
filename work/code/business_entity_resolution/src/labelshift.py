"""Label-shift (prior) correction per candidate group, estimated WITHOUT test labels.

    python labelshift.py --oof blend_v7 --test test_blend_v7     # validate on train, then estimate test priors

Why: the test has 1.6-3x more same-name / same-street / different-number "twin"
candidates than train (v05 showed that accepting more of them costs leaderboard
score). If P(features | match) is the same but the share of true matches inside
a group is lower on test, the model's train-calibrated probabilities are too high
for that group. Saerens et al. (2002) EM re-estimates the group's test prior from
the model's own posteriors and rescales them:

    p' = (pi_te/pi_tr) p / ((pi_te/pi_tr) p + ((1-pi_te)/(1-pi_tr)) (1-p))
    pi_te <- mean(p')   (iterate to convergence)

Validation (real labels): inside a train group, drop a share of the TRUE pairs to
create a known prior shift; EM must recover the new prior from probabilities only.
Only if EM recovers known shifts accurately are the test estimates trusted.
"""

import argparse

import numpy as np
import polars as pl

from config import ARTIFACT_DIR, SEED
from features import ID_COLS

GROUPS_DIR = ARTIFACT_DIR / "groups"
OOF_DIR = ARTIFACT_DIR / "oof"


def em_prior(p: np.ndarray, pi_tr: float, iters: int = 200, tol: float = 1e-7) -> tuple[float, np.ndarray]:
    """Saerens EM: estimate the target prior and return (pi_te, adjusted posteriors)."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    pi = pi_tr
    for _ in range(iters):
        a, b = pi / pi_tr, (1 - pi) / (1 - pi_tr)
        q = a * p / (a * p + b * (1 - p))
        new = float(q.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    a, b = pi / pi_tr, (1 - pi) / (1 - pi_tr)
    return pi, a * p / (a * p + b * (1 - p))


def group_expr() -> pl.Expr:
    """Same error groups as diagnose.py."""
    return (pl.when(pl.col("form_disjoint") == 1).then(pl.lit("form conflict"))
            .when((pl.col("twin_flag") == 1) & (pl.col("num_absdiff") <= 10)).then(pl.lit("near-number twin"))
            .when(pl.col("twin_flag") == 1).then(pl.lit("other-number twin"))
            .when(pl.col("t_a_empty") == 1).then(pl.lit("empty cand address"))
            .otherwise(pl.lit("rest")).alias("group"))


def load(split: str, pred_name: str) -> pl.DataFrame:
    """Predictions joined with their group labels."""
    g = pl.concat([pl.read_parquet(f) for f in sorted(GROUPS_DIR.glob(f"{split}_*.parquet"))])
    p = pl.read_parquet(OOF_DIR / f"{pred_name}.parquet")
    return p.join(g, on=ID_COLS).with_columns(group_expr())


def validate(train: pl.DataFrame) -> None:
    """Known-shift test on train: drop a share of true pairs in a group, check EM recovers the prior."""
    rng = np.random.default_rng(SEED)
    for country in sorted(train["country"].unique().to_list()):
        for grp in ("near-number twin", "other-number twin", "empty cand address"):
            d = train.filter((pl.col("country") == country) & (pl.col("group") == grp))
            if d.height < 5000:
                continue
            y, p = d["y"].to_numpy(), d["p"].to_numpy()
            pi_tr = float(y.mean())
            for keep in (1.0, 0.5, 0.25):
                mask = (y == 0) | (rng.random(len(y)) < keep)
                true_pi = float(y[mask].mean())
                est, _ = em_prior(p[mask], pi_tr)
                print(f"  [{country} {grp:18s}] keep {keep:4.2f} of true pairs: true prior {true_pi:.4f}  "
                      f"EM estimate {est:.4f}  (naive mean p {p[mask].mean():.4f})")


def estimate_test(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """Per (country, group): train prior, EM-estimated test prior, and their ratio."""
    rows = []
    tr_prior = train.group_by("group").agg(pl.col("y").mean().alias("pi_tr"), pl.col("p").mean().alias("p_tr"))
    for (country, grp), d in test.group_by(["country", "group"]):
        pi_tr = tr_prior.filter(pl.col("group") == grp)["pi_tr"][0]
        est, _ = em_prior(d["p"].to_numpy(), pi_tr)
        rows.append({"country": country, "group": grp, "n": d.height, "pi_train": round(pi_tr, 4),
                     "mean_p_test": round(float(d["p"].mean()), 4), "pi_test_EM": round(est, 4),
                     "ratio": round(est / pi_tr, 3)})
    return pl.DataFrame(rows).sort("country", "group")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="blend_v7")
    ap.add_argument("--test", default="test_blend_v7")
    a = ap.parse_args()
    tr = load("train", a.oof)
    print("VALIDATION on train (known shifts):")
    validate(tr)
    print("\nTEST prior estimates per group:")
    print(estimate_test(tr, load("test", a.test)))
