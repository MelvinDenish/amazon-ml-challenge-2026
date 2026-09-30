"""Gate v07 (blend) and the owner-probability variant against v06, OOF + label-free test counts.

Uses the k5 kernel outputs (same test candidates, groups exported by the kernel).
"""
import sys

import polars as pl

sys.path.insert(0, r"C:\Users\L Melvin Denish\Amazon_ML\work\code\business_entity_resolution\src")
from labelshift import group_expr  # noqa: E402
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05  # noqa: E402
from train import evaluate  # noqa: E402

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"


def selected(pred: pl.DataFrame, own: bool) -> pl.DataFrame:
    """Full post-processing, optionally with exclusive-owner probabilities."""
    return select_expected_f05(one_owner(exclusive_owner_prob(pred, "p") if own else pred, "p"), "p")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "oof"):
        for name in ("xgb_v6", "blend_v7"):
            oof = pl.read_parquet(f"{K5}/oof/{name}.parquet")
            for own in (False, True):
                o = exclusive_owner_prob(oof, "p") if own else oof
                r = evaluate(o, ["India", "US"])["ef05"]
                print(f"OOF {name:9s} owner_prob={own!s:5s} India {r['India']:.5f} US {r['US']:.5f} "
                      f"overall {r['overall']:.5f}", flush=True)
            del oof
    if which in ("both", "test"):
        rows = []
        for c in ("France", "India", "US"):
            g = pl.read_parquet(f"{K5}/groups/test_{c}.parquet",
                                columns=["s1_id", "cand_id", "twin_flag", "num_absdiff", "form_disjoint", "t_a_empty"]
                                ).with_columns(group_expr()).select("s1_id", "cand_id", "group")
            for name in ("xgb_v6", "blend_v7"):
                t = (pl.scan_parquet(f"{K5}/oof/test_{name}.parquet").filter(pl.col("country") == c).collect()
                     .rename({name: "p"}, strict=False).select("s1_id", "cand_id", "p"))
                n_s1 = t["s1_id"].n_unique()
                for own in (False, True):
                    m = f"{name}{'_own' if own else ''}"
                    s = selected(t, own).select("s1_id", "cand_id").join(g, on=["s1_id", "cand_id"], how="left")
                    for r in s.group_by("group").len().to_dicts():
                        rows.append({"model": m, "country": c, "group": r["group"], "v": round(r["len"] / n_s1 * 1000, 1)})
                    rows.append({"model": m, "country": c, "group": "matches_per_S1", "v": round(s.height / n_s1, 3)})
                del t
            del g
            print("done", c, flush=True)
        tab = pl.DataFrame(rows).pivot(on="model", index=["country", "group"], values="v").sort("country", "group")
        pl.Config.set_tbl_rows(40)
        pl.Config.set_tbl_width_chars(200)
        print(tab)

if __name__ == "__main__":
    main()
