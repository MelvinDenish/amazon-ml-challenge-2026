"""SageMaker training entry point: fine-tune a multilingual cross-encoder and score pair lists.

Same recipe as src/cross_encoder.py, with inputs as pre-joined text pairs (a_txt, b_txt) so the
container needs only pandas/pyarrow + the HuggingFace DLC. Channel 'data': train/eval/eval_x/test/test_x
parquet. Writes <split>_scores.parquet (s1_id, cand_id, ce) to /opt/ml/model (uploaded to S3).
Model default: xlm-roberta-large (MIT, 560M params; within the <= 8B, MIT/Apache rule).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="xlm-roberta-large")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--max-train", type=int, default=0)
a, _ = ap.parse_known_args()
D = os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data")
OUT = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
torch.manual_seed(42)
dev = "cuda"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)
print(torch.cuda.get_device_name(0), a, flush=True)


def collate(batch):
    x, y, lab = zip(*batch)
    enc = tok(list(x), list(y), truncation="longest_first", max_length=a.max_len, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(lab, dtype=torch.float32)
    return enc


@torch.no_grad()
def score(df: pd.DataFrame, bs: int = 256) -> np.ndarray:
    model.eval()
    x, y = df["a_txt"].tolist(), df["b_txt"].tolist()
    order = np.argsort([len(u) + len(v) for u, v in zip(x, y)])
    out = np.zeros(len(x), dtype=np.float32)
    for s in range(0, len(x), bs):
        idx = order[s:s + bs]
        enc = tok([x[i] for i in idx], [y[i] for i in idx], truncation="longest_first", max_length=a.max_len,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
    model.train()
    return out


tr = pd.read_parquet(f"{D}/train.parquet")
if a.max_train:
    tr = tr.sample(n=min(a.max_train, len(tr)), random_state=0)
dl = DataLoader(list(zip(tr["a_txt"], tr["b_txt"], tr["y"].astype("float32"))), batch_size=a.bs, shuffle=True,
                collate_fn=collate, num_workers=4)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
model.train()
step = 0
for _ in range(a.epochs):
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
model.save_pretrained(f"{OUT}/model")
tok.save_pretrained(f"{OUT}/model")
for split in ("eval", "eval_x", "test_x", "test"):
    df = pd.read_parquet(f"{D}/{split}.parquet")
    s = score(df)
    if "y" in df:
        yy = df["y"].to_numpy().astype(float)
        pc = 1 / (1 + np.exp(-s.astype(float)))
        print(f"{split}: logloss {-np.mean(yy * np.log(np.clip(pc, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pc, 1e-6, 1))):.4f}", flush=True)
    pd.DataFrame({"s1_id": df["s1_id"], "cand_id": df["cand_id"], "ce": s}).to_parquet(f"{OUT}/{split}_scores.parquet")
    print(f"scored {split} {len(df):,} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
