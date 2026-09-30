"""'Rescue' rule: add unselected records whose folded name equals the S1's and whose street/city words
match (token-set >= THR), when no other S1 already owns the record.

The France-empty measurement put France at ~0.955 F0.5 vs ~0.989 for US+India. France S1s have
1.90 such near-identical records per S1 but we select 1.56 (US: 1.39 vs 1.21). This rule tests the
under-selection hypothesis.

    python france_rescue.py eval <stacked_evaloof.parquet> [THR]           # held-out check (US/India labels)
    python france_rescue.py test <base_version_dir> <test_pred.parquet> <out_version> [THR]
"""
import csv
import os
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from io_utils import load_ground_truth_pairs, load_records, write_id_list_tsv  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from score import per_entity_f05  # noqa: E402

LEGAL = r"\b(sas|sasu|sarl|eurl|sa|sci|snc|ei|cie|inc|llc|ltd|co|corp|corporation|incorporated|lp|llp|pvt|private|limited|company|pc|pllc)\b"
FOLD = (pl.col("business_name").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
        .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(LEGAL, " ").str.replace_all(r"\s+", ""))
STRIP = (r"\b(rue|r|avenue|av|ave|boulevard|bd|blvd|place|pl|impasse|imp|allee|all|chemin|ch|quai|cours|crs|square|sq|"
         r"residence|res|route|rte|chaussee|passage|terrasse|de|du|des|la|le|les|l|d|no|n|appt|bis|ter)\b")
AD = (pl.col("business_address").fill_null("").str.normalize("NFKD").str.replace_all(r"\p{M}", "").str.to_lowercase()
      .str.replace_all(r"[^a-z0-9 ]", " ").str.replace_all(STRIP, " ").str.replace_all(r"\s+", " ").str.strip_chars())


def rescue(split: str, c: str, pred: pl.DataFrame, thr: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(current selection, pairs to add) for one split/country."""
    sel = select_expected_f05(one_owner(exclusive_owner_prob(pred, "p"), "p"), "p").select("s1_id", "cand_id")
    q = load_records(split, "source1", c).select(pl.col("entity_id").alias("s1_id"), FOLD.alias("f"), AD.alias("a"))
    r = pl.concat([load_records(split, s, c) for s in ("source2", "source3")]).select(
        pl.col("entity_id").alias("cand_id"), FOLD.alias("f"), AD.alias("a"))
    ob = q.filter(pl.col("f").str.len_chars() >= 4).join(r.filter(pl.col("a").str.len_chars() > 3), on="f", suffix="_t")
    sc = process.cpdist(ob["a"].to_list(), ob["a_t"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    ob = ob.with_columns(s=pl.Series(sc)).filter(pl.col("s") >= thr)
    # the rule must point to exactly ONE S1 among all S1s of the split (fair on a sampled eval universe)
    ob = ob.with_columns(n_owner=pl.len().over("cand_id")).filter(pl.col("n_owner") == 1)
    ob = ob.join(pred.select("s1_id").unique(), on="s1_id", how="semi")
    add = (ob.join(sel, on=["s1_id", "cand_id"], how="anti").join(sel.select("cand_id").unique(), on="cand_id", how="anti")
           .sort("s", descending=True).unique("cand_id", keep="first").select("s1_id", "cand_id"))
    return sel, add


def read(p: str) -> dict:
    csv.field_size_limit(10**9)
    with open(p, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        return {row[0]: [x for x in (row[1].split(",") if len(row) > 1 else []) if x] for row in rd}


def main() -> None:
    mode = sys.argv[1]
    if mode == "eval":
        ev = pl.read_parquet(sys.argv[2])
        thr = float(sys.argv[3]) if len(sys.argv) > 3 else 90
        gt = load_ground_truth_pairs()
        for c in ("India", "US"):
            o = ev.filter(pl.col("country") == c)
            sel, add = rescue("train", c, o, thr)
            s1 = o["s1_id"].unique()
            truth = gt.join(o.select("s1_id").unique(), on="s1_id")
            f0 = per_entity_f05(truth, sel, s1)["f05"].mean()
            f1 = per_entity_f05(truth, pl.concat([sel, add]).unique(), s1)["f05"].mean()
            prec = add.join(truth, on=["s1_id", "cand_id"], how="semi").height / max(add.height, 1)
            print(f"[eval {c}] thr {thr}: add {add.height / len(s1) * 1000:.1f}/1k S1, precision {prec:.1%} | "
                  f"F0.5 {f0:.5f} -> {f1:.5f} ({f1 - f0:+.5f})", flush=True)
        return
    base_dir, pred_path, out_ver = sys.argv[2], sys.argv[3], sys.argv[4]
    thr = float(sys.argv[5]) if len(sys.argv) > 5 else 90
    t = pl.scan_parquet(pred_path).filter(pl.col("country") == "France").collect()
    sel, add = rescue("test", "France", t, thr)
    n = t["s1_id"].n_unique()
    print(f"France rescue (thr {thr}): add {add.height:,} = {add.height / n * 1000:.1f}/1k S1")
    out = Path(base_dir).parent / out_ver
    out.mkdir(exist_ok=True)
    m, c = read(f"{base_dir}/matching_results.tsv"), read(f"{base_dir}/candidate_pairs.tsv")
    for s1, cid in add.iter_rows():
        m[s1].append(cid)
        if cid not in c[s1]:
            c[s1].append(cid)
    ids = list(m.keys())
    write_id_list_tsv(ids, pl.DataFrame([(k, v) for k, vs in m.items() for v in vs], schema=["s1_id", "cand_id"], orient="row"),
                      out / "matching_results.tsv", "matched_entity_ids")
    write_id_list_tsv(ids, pl.DataFrame([(k, v) for k, vs in c.items() for v in vs], schema=["s1_id", "cand_id"], orient="row"),
                      out / "candidate_pairs.tsv", "candidate_entity_ids")
    print("wrote", out)


if __name__ == "__main__":
    main()
