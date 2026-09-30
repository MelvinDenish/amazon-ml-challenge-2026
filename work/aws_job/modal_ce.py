"""Modal GPU job: fine-tune a multilingual cross-encoder and score the CE pair lists.

    modal run --detach modal_ce.py --tag xlmrl --model xlm-roberta-large --urls urls_xlmrl.json

Data comes in and results go out through presigned S3 URLs (no credentials leave the laptop).
Inputs: train_full (1.47M real train pairs, eval S1s excluded), optional train_fr (France test pairs
with confident labels + synthetic French siblings), eval / eval_x (held-out, labelled), test / test_x.
Models: xlm-roberta-large (MIT, 560M) or microsoft/mdeberta-v3-base (MIT, 278M): within <= 8B, MIT/Apache.
"""
import json

import modal

app = modal.App("bedrock-ce")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "transformers==4.44.2", "pandas", "pyarrow", "requests", "sentencepiece",
                      "protobuf", "tiktoken"))


@app.function(image=image, gpu="H100", timeout=6 * 3600, cpu=8, memory=65536)
def run(tag: str, model_name: str, urls: dict, lr: float, bs: int, use_fr: bool, epochs: int = 1) -> str:
    """Fine-tune, score every split, upload each score file to its presigned PUT URL."""
    import io
    import time

    import numpy as np
    import pandas as pd
    import requests
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    t0 = time.time()

    def get(name):
        r = requests.get(urls["get"][name], timeout=600)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))

    tr = get("train_full")
    if use_fr:
        tr = pd.concat([tr, get("train_fr")], ignore_index=True)
    print(f"[{tag}] {torch.cuda.get_device_name(0)} train pairs {len(tr):,} ({time.time() - t0:.0f}s)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to("cuda")

    def collate(batch):
        x, y, lab = zip(*batch)
        enc = tok(list(x), list(y), truncation="longest_first", max_length=128, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(lab, dtype=torch.float32)
        return enc

    @torch.no_grad()
    def score(df, b=1024):
        model.eval()
        x, y = df["a_txt"].tolist(), df["b_txt"].tolist()
        order = np.argsort([len(u) + len(v) for u, v in zip(x, y)])
        out = np.zeros(len(x), dtype=np.float32)
        for s in range(0, len(x), b):
            idx = order[s:s + b]
            enc = tok([x[i] for i in idx], [y[i] for i in idx], truncation="longest_first", max_length=128,
                      padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()
        model.train()
        return out

    dl = DataLoader(list(zip(tr["a_txt"], tr["b_txt"], tr["y"].astype("float32"))), batch_size=bs, shuffle=True,
                    collate_fn=collate, num_workers=6)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    for _ in range(epochs):
        for batch in dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(**batch).logits.squeeze(-1).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            if step % 1000 == 0:
                print(f"[{tag}] step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    report = []
    for split in [k for k in ("eval", "eval_x", "test_x", "test") if k in urls["put"]]:
        df = get(split)
        s = score(df)
        if "y" in df:
            yy = df["y"].to_numpy().astype(float)
            pc = 1 / (1 + np.exp(-s.astype(float)))
            ll = -np.mean(yy * np.log(np.clip(pc, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pc, 1e-6, 1)))
            report.append(f"{split} logloss {ll:.4f}")
            print(f"[{tag}] {split} logloss {ll:.4f}", flush=True)
        buf = io.BytesIO()
        pd.DataFrame({"s1_id": df["s1_id"], "cand_id": df["cand_id"], "ce": s}).to_parquet(buf)
        requests.put(urls["put"][split], data=buf.getvalue(), timeout=600).raise_for_status()
        print(f"[{tag}] uploaded {split} {len(df):,} {time.time() - t0:.0f}s", flush=True)
    return f"{tag} done: " + "; ".join(report)


@app.local_entrypoint()
def main(tag: str, model: str, urls: str, lr: float = 1e-5, bs: int = 64, fr: int = 0, epochs: int = 1):
    u = json.load(open(urls))
    print(run.remote(tag, model, u, lr, bs, bool(fr), epochs))
