"""Kaggle GPU job (Bedrock ER): France adaptation of the XLM-R cross-encoder (continued from
bedrock-k9c-xlmr, lr 1e-5, 1 epoch) on ce_train_fr (300k original train pairs + France test pairs with
confident labels from the current stack + synthetic French sibling negatives, see ce_prep_france.py).
Scores: ce_eval (held-out US/India, to check nothing degrades) and the France test pairs of ce_test and
ce_test_x. Outputs ce_eval_scores_fr.parquet, ce_test_scores_fr.parquet, test_x_scores_fr.parquet."""
import glob
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = os.path.dirname([p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "bedrock-k9f-xlmr2" in p][0])
MAX_LEN, BS, LR = 128, 64, 1e-5
torch.manual_seed(44)
dev = "cuda"
data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
ce = os.path.dirname(glob.glob("/kaggle/input/**/ce_train_fr.parquet", recursive=True)[0])


def texts(split: str) -> pl.DataFrame:
    parts = [pl.read_parquet(f"{data}/{split}_{s}.parquet", columns=["entity_id", "business_name", "business_address"])
             for s in ("source1", "source2", "source3")]
    return pl.concat(parts)


def fmt(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("entity_id", pl.concat_str([pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")],
                                                separator=" ; ").str.slice(0, 400).alias("txt")).unique("entity_id", keep="first")


def attach(pairs: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    return (pairs.join(tx.rename({"entity_id": "s1_id", "txt": "a_txt"}), on="s1_id", how="left")
            .join(tx.rename({"entity_id": "cand_id", "txt": "b_txt"}), on="cand_id", how="left")
            .with_columns(pl.col("a_txt", "b_txt").fill_null("")))


t0 = time.time()
tx_tr, tx_te = fmt(texts("train")), fmt(texts("test"))
tx_all = pl.concat([tx_tr, tx_te, fmt(pl.read_parquet(f"{ce}/syn_records_fr.parquet"))]).unique("entity_id", keep="first")
tr = attach(pl.read_parquet(f"{ce}/ce_train_fr.parquet"), tx_all)
print(f"train pairs {tr.height:,}  missing text a/b {int((tr['a_txt'] == '').sum())}/{int((tr['b_txt'] == '').sum())}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev)


def collate(batch):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.float32)
    return enc


@torch.no_grad()
def score(df: pl.DataFrame, bs: int = 512) -> np.ndarray:
    model.eval()
    a, b = df["a_txt"].to_list(), df["b_txt"].to_list()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.zeros(len(a), dtype=np.float32)
    for s in range(0, len(a), bs):
        idx = order[s:s + bs]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation="longest_first", max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
    model.train()
    return out


opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
dl = DataLoader(list(zip(tr["a_txt"].to_list(), tr["b_txt"].to_list(), tr["y"].cast(pl.Float32).to_list())),
                batch_size=BS, shuffle=True, collate_fn=collate, num_workers=2)
steps = len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.02 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
model.train()
for i, batch in enumerate(dl, 1):
    batch = {k: v.to(dev) for k, v in batch.items()}
    y = batch.pop("labels")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = lossf(model(**batch).logits.squeeze(-1).float(), y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sch.step()
    if i % 1000 == 0:
        print(f"step {i}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
ev = attach(pl.read_parquet(f"{ce}/ce_eval.parquet"), tx_tr)
ev_s = score(ev)
y = ev["y"].to_numpy().astype(float)
pc = 1 / (1 + np.exp(-ev_s.astype(float)))
print(f"eval (held-out US/India) CE logloss {-np.mean(y * np.log(np.clip(pc, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pc, 1e-6, 1))):.4f}", flush=True)
ev.select("s1_id", "cand_id").with_columns(pl.Series("ce", ev_s)).write_parquet("/kaggle/working/ce_eval_scores_fr.parquet")
for name, out in (("ce_test.parquet", "ce_test_scores_fr.parquet"), ("ce_test_x.parquet", "test_x_scores_fr.parquet")):
    te = attach(pl.read_parquet(f"{ce}/{name}").filter(pl.col("country") == "France"), tx_te)
    te.select("s1_id", "cand_id").with_columns(pl.Series("ce", score(te))).write_parquet(f"/kaggle/working/{out}")
    print(f"scored {name} France {te.height:,} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
