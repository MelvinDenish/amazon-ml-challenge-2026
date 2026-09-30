"""Sibling-cell prior correction (label shift inside structurally defined cells).

Evidence (sibling_cells.log): pairs unaffected by distractors ('no added word, same number') occur at
the SAME rate per S1 on train and test (US 1351 vs 1344 per 1k S1): true records are generated the
same way. The extra ~1.1 records per S1 on test land in 'differs' cells ('<name> + word' or
'+ legal form' at another house number next to an anchored cluster): US 130 -> 498 per 1k S1.
So inside a cell c the share of true pairs on test is lower:
    pi_test(c) = pi_train(c) * n_train(c) / n_test(c)     (true pairs per S1 unchanged)
and the train-calibrated posterior is corrected by Bayes (Saerens et al. 2002):
    odds'(pair) = odds(pair) * [pi_test/(1-pi_test)] / [pi_train/(1-pi_train)]
US and India use the count ratio (shown next to EM). France (no labels) uses EM with the pooled
train prior. Ratios are capped at 1 (only down-weighting). Cells: added-word type x number relation
(same / near <=10 / far / missing) x anchor.

    python cell_shift.py <train_oof> <test_pred> <out_test_pred>
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_records  # noqa: E402
from labelshift import em_prior  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402

LEGAL = ["inc", "llc", "ltd", "co", "corp", "corporation", "incorporated", "lp", "llp", "pvt", "private", "limited",
         "company", "plc", "pllc", "pc", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "cie", "l", "c", "p"]
TOK = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
       .str.replace_all(r"[^a-z0-9 ]", " ").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique())
NUM = pl.col("business_address").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1")
MIN_N = 2000
EXCESS = 1.3  # correct a cell only if test has >= 30% more of it per S1 than train (a real excess)


def cells(pred: pl.DataFrame, split: str, c: str) -> pl.DataFrame:
    """Per plausible pair (p > 0.02): its cell label."""
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), TOK.alias("qt"), NUM.alias("qn"))
    t = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), TOK.alias("tt"), NUM.alias("tn"))
    d = pred.filter(pl.col("p") > 0.02).join(q, on="s1_id").join(t, on="cand_id", how="left")
    d = d.with_columns(same=(pl.col("qn").is_not_null() & (pl.col("qn") == pl.col("tn"))).fill_null(False),
                       add=pl.col("tt").list.set_difference(pl.col("qt")),
                       diff=(pl.col("qn").cast(pl.Int64, strict=False) - pl.col("tn").cast(pl.Int64, strict=False)).abs())
    d = d.with_columns(anchor_n=(pl.col("same") & (pl.col("p") >= 0.9)).cast(pl.Int32).sum().over("s1_id"))
    self_anchor = (pl.col("same") & (pl.col("p") >= 0.9)).cast(pl.Int32)
    return d.with_columns(pl.concat_str([
        pl.when(pl.col("add").list.len() == 0).then(pl.lit("none"))
        .when(pl.col("add").list.eval(pl.element().is_in(LEGAL)).list.all()).then(pl.lit("legal"))
        .otherwise(pl.lit("word")),
        pl.when(pl.col("qn").is_null() | pl.col("tn").is_null()).then(pl.lit("missing"))
        .when(pl.col("same")).then(pl.lit("same"))
        .when(pl.col("diff") <= 10).then(pl.lit("near")).otherwise(pl.lit("far")),
        pl.when((pl.col("anchor_n") - self_anchor) >= 1).then(pl.lit("anch")).otherwise(pl.lit("free")),
    ], separator="|").alias("cell")).drop("qt", "tt", "qn", "tn", "add", "same", "diff", "anchor_n")


def main() -> None:
    oof_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    oof = pl.read_parquet(oof_path)
    tr_stats = {}
    for c in ("India", "US"):
        o = oof.filter(pl.col("country") == c)
        n = o["s1_id"].n_unique()
        tr_stats[c] = cells(o.select("s1_id", "cand_id", "y", "p"), "train", c).group_by("cell").agg(
            pl.len().alias("n_tr"), (pl.len() / n * 1000).alias("per1k_tr"), pl.col("y").mean().alias("pi_tr"))
        print(c, "train cells done", flush=True)
    del oof
    pooled = pl.concat([tr_stats[c].with_columns((pl.col("pi_tr") * pl.col("n_tr")).alias("pos")) for c in tr_stats]
                       ).group_by("cell").agg(pl.col("n_tr").sum(), pl.col("pos").sum()).with_columns(
        (pl.col("pos") / pl.col("n_tr")).alias("pi_tr"), pl.lit(None, dtype=pl.Float64).alias("per1k_tr"))
    parts = []
    us_tab = None
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    for c in ("US", "India", "France"):
        t = pl.scan_parquet(test_path).filter(pl.col("country") == c).collect()
        if "p" not in t.columns:
            t = t.rename({[x for x in t.columns if x not in ("s1_id", "cand_id", "country")][0]: "p"})
        n = t["s1_id"].n_unique()
        tc = cells(t.select("s1_id", "cand_id", "p"), "test", c)
        ref = tr_stats.get(c, pooled)
        rows = []
        for (cell,), g in tc.group_by("cell"):
            per1k_te = g.height / n * 1000
            r = ref.filter(pl.col("cell") == cell)
            if r.height == 0 or r["n_tr"][0] < MIN_N or g.height < MIN_N:
                continue
            pi_tr = float(r["pi_tr"][0])
            if not 0.001 < pi_tr < 0.999:
                continue
            pi_em, _ = em_prior(g["p"].to_numpy(), pi_tr)
            pi_ratio, lr = np.nan, 1.0
            differs = cell.split("|")[1] in ("near", "far")
            if c in tr_stats:
                excess = per1k_te / float(r["per1k_tr"][0])
                pi_ratio = pi_tr / excess
                if differs and excess >= EXCESS:
                    lr = (pi_ratio / (1 - pi_ratio)) / (pi_tr / (1 - pi_tr))
            elif differs:
                us = us_tab.filter(pl.col("cell") == cell)
                base = max(float(tr_stats[k].filter(pl.col("cell") == cell)["per1k_tr"].max() or 0) for k in tr_stats)
                if us.height and base > 0 and per1k_te / base >= EXCESS:
                    lr = float(us["odds_mult"][0])
            rows.append({"cell": cell, "per1k_tr": None if c not in tr_stats else round(float(r["per1k_tr"][0]), 1),
                         "per1k_te": round(per1k_te, 1), "pi_tr": round(pi_tr, 3), "mean_p_te": round(float(g["p"].mean()), 3),
                         "pi_ratio": None if c not in tr_stats else round(pi_ratio, 3), "pi_em": round(pi_em, 3),
                         "odds_mult": round(lr, 3)})
        tab = pl.DataFrame(rows).sort("cell")
        if c == "US":
            us_tab = tab
        print(f"\n######## TEST {c}  (S1 {n:,})")
        print(tab)
        mult = tc.join(tab.select("cell", "odds_mult"), on="cell", how="left").select(
            "s1_id", "cand_id", pl.col("odds_mult").fill_null(1.0))
        tv = t.join(mult, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("odds_mult").fill_null(1.0))
        o = pl.col("p").cast(pl.Float64).clip(1e-7, 1 - 1e-7)
        o = o / (1 - o) * pl.col("odds_mult")
        tv = tv.with_columns((o / (1 + o)).cast(pl.Float32).alias("p_new"))
        b = select_expected_f05(one_owner(exclusive_owner_prob(t, "p"), "p"), "p").select("s1_id", "cand_id")
        tn = tv.select("s1_id", "cand_id", "country", pl.col("p_new").alias("p"))
        a = select_expected_f05(one_owner(exclusive_owner_prob(tn, "p"), "p"), "p").select("s1_id", "cand_id")
        rem = b.join(a, on=["s1_id", "cand_id"], how="anti")
        add = a.join(b, on=["s1_id", "cand_id"], how="anti")
        mp = rem.join(tn, on=["s1_id", "cand_id"])["p"].mean()
        own = one_owner(exclusive_owner_prob(tn, "p"), "p").select("s1_id", "cand_id", "p")

        def exp_f(sel: pl.DataFrame) -> float:
            """Expected macro F0.5 of a selection if the corrected owned probabilities were the truth."""
            d = own.join(sel.with_columns(pl.lit(1).alias("s")), on=["s1_id", "cand_id"], how="left").with_columns(
                pl.col("s").fill_null(0))
            g = d.group_by("s1_id").agg(pl.col("p").sum().alias("et"), (pl.col("p") * pl.col("s")).sum().alias("tp"),
                                        pl.col("s").sum().alias("m"), (1 - pl.col("p")).clip(1e-6, 1).log().sum().exp().alias("p0"))
            f = pl.when(pl.col("m") == 0).then(pl.col("p0")).otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("et") / 0.985 + pl.col("m")))
            return float(g.select(f.mean()).item())
        print(f"   expected F0.5 under corrected p: old selection {exp_f(b):.5f} -> new {exp_f(a):.5f}", flush=True)
        print(f"TEST {c}: selections removed {rem.height / n * 1000:.1f}/1k S1, added {add.height / n * 1000:.1f}/1k; "
              f"matches/S1 {b.height / n:.3f} -> {a.height / n:.3f}; mean corrected p of removed {(mp or 0):.3f}", flush=True)
        parts.append(tn)
        del t, tc, tv
    pl.concat(parts).write_parquet(out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
