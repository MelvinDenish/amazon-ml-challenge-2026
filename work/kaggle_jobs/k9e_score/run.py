"""Kaggle GPU job (Bedrock ER): score the risky CONFIDENT pairs (GBM p >= 0.995 in twin / legal-form /
empty-address groups) with the three already fine-tuned cross-encoders (no training).
Inputs: saved models from kernels bedrock-k9-ce (v1), bedrock-k9b-ce2 (v2), bedrock-k9c-xlmr (xlmr);
pair lists ce_eval_x / ce_test_x from dataset bedrock-ce. Outputs <split>_x_scores_<tag>.parquet."""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
ce = os.path.dirname(glob.glob("/kaggle/input/**/ce_eval_x.parquet", recursive=True)[0])
MODELS = {}
for tag, slug in (("v1", "bedrock-k9-ce"), ("v2", "bedrock-k9b-ce2"), ("xlmr", "bedrock-k9c-xlmr")):
    cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if slug + "/" in p or slug in p.split("/")]
    MODELS[tag] = os.path.dirname(cfg[0])
print(MODELS, flush=True)


def texts(split: str) -> pl.DataFrame:
    parts = [pl.read_parquet(f"{data}/{split}_{s}.parquet", columns=["entity_id", "business_name", "business_address"])
             for s in ("source1", "source2", "source3")]
    return pl.concat(parts).select("entity_id", pl.concat_str(
        [pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")], separator=" ; ").str.slice(0, 400).alias("txt"))


def attach(pairs: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    return (pairs.join(tx.rename({"entity_id": "s1_id", "txt": "a_txt"}), on="s1_id", how="left")
            .join(tx.rename({"entity_id": "cand_id", "txt": "b_txt"}), on="cand_id", how="left")
            .with_columns(pl.col("a_txt", "b_txt").fill_null("")))


t0 = time.time()
sets = {"eval": attach(pl.read_parquet(f"{ce}/ce_eval_x.parquet"), texts("train")),
        "test": attach(pl.read_parquet(f"{ce}/ce_test_x.parquet"), texts("test"))}
for tag, path in MODELS.items():
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
    for split, df in sets.items():
        a, b = df["a_txt"].to_list(), df["b_txt"].to_list()
        order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
        out = np.zeros(len(a), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(a), 512):
                idx = order[s:s + 512]
                enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation="longest_first", max_length=128,
                          padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
        df.select("s1_id", "cand_id").with_columns(pl.Series("ce", out)).write_parquet(f"/kaggle/working/{split}_x_scores_{tag}.parquet")
        print(f"{tag} {split} {len(a):,} pairs  mean logit {out.mean():.3f}  {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
print("done")
