"""CE v2 training pairs: 4x more data. All train S1s with honest OOF p (xgb_v6 or xgb_v7b samples)
EXCEPT the CE-eval S1s (v07-blend S1s inside the v08 sample), so the stacking evaluation stays clean."""
import polars as pl

K5 = r"C:\Users\L Melvin Denish\Amazon_ML\work\artifacts\k5_out\artifacts"
OUT = r"C:\Users\L Melvin Denish\Amazon_ML\work\kaggle_jobs\bedrock_ce"
ev = pl.read_parquet(f"{OUT}/ce_eval.parquet").select("s1_id").unique()
blend_s1 = pl.scan_parquet(f"{K5}/oof/blend_v7.parquet").select("s1_id").unique().collect()
eval_s1 = blend_s1.filter((pl.col("s1_id").hash(seed=45) % 10_000) < 5000)  # all eval-universe S1s, not only routed
v6 = pl.read_parquet(f"{K5}/oof/xgb_v6.parquet").select("s1_id", "cand_id", "country", "y", "p")
v7 = pl.read_parquet(f"{K5}/oof/xgb_v7b.parquet").select("s1_id", "cand_id", "country", "y", "p")
v7 = v7.join(v6.select("s1_id").unique(), on="s1_id", how="anti")
pool = pl.concat([v6, v7]).join(eval_s1, on="s1_id", how="anti")
print("S1s", pool["s1_id"].n_unique(), "pairs", pool.height)
band = (pl.col("p") > 0.02) & (pl.col("p") < 0.98)
tr = pl.concat([
    pool.filter(band),
    pool.filter((pl.col("p") >= 0.98) & (pl.col("y") == 0)),
    pool.filter((pl.col("p") <= 0.02) & (pl.col("y") == 1)),
    pool.filter((pl.col("p") >= 0.98) & (pl.col("y") == 1)).sample(200_000, seed=1),
    pool.filter((pl.col("p") <= 0.02) & (pl.col("y") == 0)).sample(200_000, seed=2),
]).unique(["s1_id", "cand_id"])
print("CE v2 train pairs", tr.height, "pos rate", round(float(tr["y"].mean()), 3), tr.group_by("country").len().to_dicts())
tr.write_parquet(f"{OUT}/ce_train2.parquet")
