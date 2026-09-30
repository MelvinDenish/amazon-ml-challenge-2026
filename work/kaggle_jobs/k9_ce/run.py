"""Kaggle GPU job (Bedrock ER): multilingual cross-encoder pilot (review item C).

Model: distilbert-base-multilingual-cased (Apache-2.0, 134M params; allowed: <= 8B,
Apache/MIT). Fine-tuned ONLY on the supplied train labels, on pairs from S1s outside
the v08 sample (uncertain band + confident mistakes + easy controls, see ce_prep.py).
Each record is serialised from its RAW name and address (native scripts kept, which
the fuzzy features cannot compare). The job scores:
  ce_eval  pairs of held-out S1s (for stacking + full-pipeline evaluation on the laptop)
  ce_test  uncertain test pairs (0.005 < p < 0.995 under v07)
Outputs /kaggle/working/ce_eval_scores.parquet, ce_test_scores.parquet, train log.
"""
import glob
import math
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "distilbert-base-multilingual-cased"
MAX_LEN, BS, LR, EPOCHS = 128, 64, 4e-5, 1
torch.manual_seed(42)
dev = "cuda"
print(torch.cuda.get_device_name(0), flush=True)

data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
ce = os.path.dirname(glob.glob("/kaggle/input/**/ce_train.parquet", recursive=True)[0])


def texts(split: str) -> pl.DataFrame:
    """entity_id -> 'name ; address' from the raw records of one split."""
    parts = [pl.read_parquet(f"{data}/{split}_{s}.parquet", columns=["entity_id", "business_name", "business_address"])
             for s in ("source1", "source2", "source3")]
    return pl.concat(parts).select(
        "entity_id",
        pl.concat_str([pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")],
                      separator=" ; ").str.slice(0, 400).alias("txt"))


def attach(pairs: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    """Add a_txt (S1) and b_txt (candidate)."""
    return (pairs.join(tx.rename({"entity_id": "s1_id", "txt": "a_txt"}), on="s1_id", how="left")
            .join(tx.rename({"entity_id": "cand_id", "txt": "b_txt"}), on="cand_id", how="left")
            .with_columns(pl.col("a_txt", "b_txt").fill_null(""))
            )


tok = AutoTokenizer.from_pretrained(MODEL)


def collate(batch):
    a, b, y = zip(*batch)
    enc = tok(list(a), list(b), truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.float32)
    return enc


t0 = time.time()
tx = texts("train")
tr = attach(pl.read_parquet(f"{ce}/ce_train.parquet"), tx)
ev = attach(pl.read_parquet(f"{ce}/ce_eval.parquet"), tx)
del tx
# small monitoring split by S1 hash
mon = (tr["s1_id"].hash(seed=7) % 100 < 3).to_numpy()
fit, val = tr.filter(~pl.Series(mon)), tr.filter(pl.Series(mon))
print(f"train pairs {fit.height:,} monitor {val.height:,} eval {ev.height:,}  prep {time.time() - t0:.0f}s", flush=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
ds = list(zip(fit["a_txt"].to_list(), fit["b_txt"].to_list(), fit["y"].cast(pl.Float32).to_list()))
dl = DataLoader(ds, batch_size=BS, shuffle=True, collate_fn=collate, num_workers=2)
steps = EPOCHS * len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()


@torch.no_grad()
def score(df: pl.DataFrame, bs: int = 512) -> np.ndarray:
    """Logits for all pairs; length-sorted batches with dynamic padding for speed."""
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


def report(tag: str, df: pl.DataFrame, logit: np.ndarray) -> None:
    """Logloss/accuracy of the CE alone and of the model p alone on the same pairs."""
    y = df["y"].to_numpy().astype(np.float64)
    pc = 1 / (1 + np.exp(-logit.astype(np.float64)))
    pm = df["p"].to_numpy().astype(np.float64)
    ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    print(f"  [{tag}] n={len(y):,} CE logloss {ll(pc):.4f} acc {np.mean((pc > .5) == y):.4f} | "
          f"GBM logloss {ll(pm):.4f} acc {np.mean((pm > .5) == y):.4f}", flush=True)


model.train()
step = 0
for ep in range(EPOCHS):
    for batch in dl:
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
        step += 1
        if step % 500 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % 2000 == 0 or step == steps:
            report("monitor", val, score(val))

ev_logit = score(ev)
report("eval (held-out S1s)", ev, ev_logit)
ev.select("s1_id", "cand_id").with_columns(pl.Series("ce", ev_logit)).write_parquet("/kaggle/working/ce_eval_scores.parquet")
print(f"eval scored {time.time() - t0:.0f}s", flush=True)
model.save_pretrained("/kaggle/working/ce_model")
tok.save_pretrained("/kaggle/working/ce_model")

tx = texts("test")
te = attach(pl.read_parquet(f"{ce}/ce_test.parquet"), tx)
del tx
out = []
for s in range(0, te.height, 500_000):
    part = te.slice(s, 500_000)
    out.append(part.select("s1_id", "cand_id").with_columns(pl.Series("ce", score(part))))
    print(f"test {s + part.height:,}/{te.height:,} {time.time() - t0:.0f}s", flush=True)
pl.concat(out).write_parquet("/kaggle/working/ce_test_scores.parquet")
print("done", flush=True)
