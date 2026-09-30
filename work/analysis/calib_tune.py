"""Item 5: tune the final calibration for macro F0.5 on the stacked held-out OOF, cross-checked on halves.
Logit scale a and shift b: p' = sigmoid(a*logit(p) + b); chosen on one S1 half, reported on the other."""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from postprocess import exclusive_owner_prob  # noqa: E402
from train import evaluate  # noqa: E402

oof = pl.read_parquet(sys.argv[1]).with_columns((pl.col("s1_id").hash(seed=21) % 2).alias("h"))


def f(df: pl.DataFrame, a: float, b: float) -> float:
    p = pl.col("p").cast(pl.Float64).clip(1e-7, 1 - 1e-7)
    x = df.with_columns((1 / (1 + (-(a * (p / (1 - p)).log() + b)).exp())).cast(pl.Float32).alias("p"))
    tot, n = 0.0, 0
    for c in ("India", "US"):
        s = x.filter(pl.col("country") == c)
        k = s["s1_id"].n_unique()
        tot += evaluate(exclusive_owner_prob(s, "p"), [c])["ef05"][c] * k
        n += k
    return tot / n


grid = [(1.0, 0.0), (1.0, -0.3), (1.0, 0.3), (0.85, 0.0), (1.15, 0.0), (1.0, -0.6), (1.0, 0.6)]
for h in (0, 1):
    tune, hold = oof.filter(pl.col("h") == h), oof.filter(pl.col("h") != h)
    res = {g: f(tune, *g) for g in grid}
    best = max(res, key=res.get)
    print(f"half {h}: tune scores {({k: round(v, 5) for k, v in res.items()})}")
    print(f"   best {best}: held-out {f(hold, *best):.5f} vs identity {f(hold, 1.0, 0.0):.5f}", flush=True)
