"""Multilingual cross-encoder re-scorer for uncertain candidate pairs.

    python cross_encoder.py --model distilbert-base-multilingual-cased --train ce_train2.parquet
    python cross_encoder.py --model xlm-roberta-base --lr 2e-5 --train ce_train2.parquet --tag xlmr

Why: the GBM sees fuzzy-similarity numbers, not the words themselves. The test set is full of
'sibling' distractors ('<name> Holdings / Riverside / Développement' at a nearby number) and of
native-script names; a transformer reading both raw records can tell those apart.
Pretrained models used: distilbert-base-multilingual-cased (Apache-2.0, 134M) and
xlm-roberta-base (MIT, 278M): both within the <= 8B / MIT-Apache rule. Fine-tuned ONLY on the
supplied train labels.

Inputs (artifacts/ce/): ce_train*.parquet (s1_id, cand_id, y), ce_eval.parquet (held-out S1s,
for stacking), ce_test.parquet (uncertain test pairs). Each record is serialised as
'<raw name> ; <raw address>'. Outputs artifacts/ce/ce_{eval,test}_scores[_tag].parquet (logits).
"""

import argparse
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from config import ARTIFACT_DIR
from io_utils import load_records

CE_DIR = ARTIFACT_DIR / "ce"


def texts(split: str) -> pl.DataFrame:
    """entity_id -> 'name ; address' from the raw records of one split."""
    parts = [load_records(split, s).select("entity_id", "business_name", "business_address")
             for s in ("source1", "source2", "source3")]
    return pl.concat(parts).select(
        "entity_id",
        pl.concat_str([pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")],
                      separator=" ; ").str.slice(0, 400).alias("txt"))


def attach(pairs: pl.DataFrame, tx: pl.DataFrame) -> pl.DataFrame:
    """Add a_txt (S1) and b_txt (candidate) to a pair table."""
    return (pairs.join(tx.rename({"entity_id": "s1_id", "txt": "a_txt"}), on="s1_id", how="left")
            .join(tx.rename({"entity_id": "cand_id", "txt": "b_txt"}), on="cand_id", how="left")
            .with_columns(pl.col("a_txt", "b_txt").fill_null("")))


def main() -> None:
    """Fine-tune the cross-encoder, then score the eval and test pair lists."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    ap.add_argument("--train", default="ce_train2.parquet")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(42)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)

    def collate(batch):
        x, y, lab = zip(*batch)
        enc = tok(list(x), list(y), truncation="longest_first", max_length=a.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(lab, dtype=torch.float32)
        return enc

    t0 = time.time()
    tx = texts("train")
    tr = attach(pl.read_parquet(CE_DIR / a.train), tx)
    ev = attach(pl.read_parquet(CE_DIR / "ce_eval.parquet"), tx)
    del tx
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1).to(dev)

    @torch.no_grad()
    def score(df: pl.DataFrame, bs: int = 512) -> np.ndarray:
        """Logits for all pairs; length-sorted batches with dynamic padding."""
        model.eval()
        x, y = df["a_txt"].to_list(), df["b_txt"].to_list()
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

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    ds = list(zip(tr["a_txt"].to_list(), tr["b_txt"].to_list(), tr["y"].cast(pl.Float32).to_list()))
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2)
    steps = a.epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
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
            if step % 1000 == 0:
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    sfx = f"_{a.tag}" if a.tag else ""
    ev.select("s1_id", "cand_id").with_columns(pl.Series("ce", score(ev))).write_parquet(CE_DIR / f"ce_eval_scores{sfx}.parquet")
    tx = texts("test")
    te = attach(pl.read_parquet(CE_DIR / "ce_test.parquet"), tx)
    del tx
    out = [te.slice(s, 500_000).select("s1_id", "cand_id").with_columns(pl.Series("ce", score(te.slice(s, 500_000))))
           for s in range(0, te.height, 500_000)]
    pl.concat(out).write_parquet(CE_DIR / f"ce_test_scores{sfx}.parquet")
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
