"""Test-side count gate for post-hoc corrections: selections per 1k S1 by suspicious group.

Compares test selections of several prediction files (all with owner prob + expected-F0.5) against
the TRAIN OOF reference of the same pipeline, per country. A correction that is right moves the
test counts TOWARD the train level (e.g. US near-number twins 102 -> ~51) and not below it.

    python shift_gate.py <train_oof> <test_pred_1> [<test_pred_2> ...]
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from labelshift import group_expr  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
GCOLS = ["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty"]


def counts(pred: pl.DataFrame, split: str, c: str, tag: str) -> list[dict]:
    """Selections per 1k S1 by group (+ matches per S1, and precision when labels exist)."""
    g = pl.read_parquet(f"{K5}/groups/{split}_{c}.parquet", columns=GCOLS).with_columns(group_expr()).select(
        "s1_id", "cand_id", "group")
    n = pred["s1_id"].n_unique()
    sel = select_expected_f05(one_owner(exclusive_owner_prob(pred, "p"), "p"), "p")
    s = sel.join(g, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("group").fill_null("rest"))
    aggs = [pl.len().alias("k")] + ([pl.col("y").mean().alias("prec")] if "y" in s.columns else [])
    out = []
    for r in s.group_by("group").agg(aggs).to_dicts():
        v = f"{r['k'] / n * 1000:.1f}" + (f" ({r['prec']:.2f})" if "prec" in r else "")
        out.append({"country": c, "group": r["group"], "model": tag, "v": v})
    out.append({"country": c, "group": "matches_per_S1", "model": tag, "v": f"{sel.height / n:.3f}"})
    return out


def main() -> None:
    oof_path, tests = sys.argv[1], sys.argv[2:]
    rows = []
    oof = pl.read_parquet(oof_path)
    for c in ("India", "US"):
        rows += counts(oof.filter(pl.col("country") == c), "train", c, "TRAIN ref (precision)")
    del oof
    for c in ("France", "India", "US"):
        for tp in tests:
            t = pl.scan_parquet(tp).filter(pl.col("country") == c).collect()
            if "p" not in t.columns:
                t = t.rename({[x for x in t.columns if x not in ("s1_id", "cand_id", "country")][0]: "p"})
            tag = tp.replace("\\", "/").rsplit("/", 1)[-1].replace(".parquet", "")
            rows += counts(t.select("s1_id", "cand_id", "country", "p"), "test", c, tag)
        print("done", c, flush=True)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(250)
    print(pl.DataFrame(rows).pivot(on="model", index=["country", "group"], values="v").sort("country", "group"))


if __name__ == "__main__":
    main()
