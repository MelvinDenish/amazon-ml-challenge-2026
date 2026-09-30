"""Stage-2 cluster re-scorer: uses stage-1 probabilities to judge each candidate
against the S1's *confident* cluster members.

    python stage2.py --stage1 xgb_v4x --name s2_v5          # build features, train (GPU), OOF report
    python stage2.py --stage1 xgb_v4x --test-only           # test shards (needs test_<stage1> preds)

Motivation (measured on v4x OOF): the largest remaining error is missed true
matches whose candidate record has an EMPTY address (~44 per 1k S1) - stage 1
has no address evidence for them. But an entity's records form a cluster: if
the S1's confident members (p > 0.8) all carry name X, an empty-address record
named exactly X is very likely a member. Likewise a record whose house number
disagrees with the consensus of the confident members is likely a twin.

Leakage control: train-side features use stage-1 OUT-OF-FOLD probabilities, and
stage 2 reuses the same S1-hash folds.
"""

import argparse
import time

import numpy as np
import polars as pl
import xgboost as xgb

from config import ARTIFACT_DIR, SEED
from dataset import load_country
from features import ID_COLS
from postprocess import ranked
from train import MODEL_DIR, OOF_DIR, evaluate, load_train_features
from train_gpu import XGB_PARAMS

S2_DIR = ARTIFACT_DIR / "feats_s2"
CONF = 0.8


def cluster_features(pred: pl.DataFrame, t_attrs: pl.DataFrame, q_attrs: pl.DataFrame) -> pl.DataFrame:
    """Stage-2 features for pairs (s1_id, cand_id, p) given record attributes.

    t_attrs: cand_id, t_num, t_name (normalised core name); q_attrs: s1_id, q_num.
    """
    x = pred.select(ID_COLS + ["p"]).join(t_attrs, on="cand_id", how="left").join(q_attrs, on="s1_id", how="left")
    x = ranked(x, "p", "s1_id", "cand_id", "p_rank")
    top2 = x.group_by("cand_id").agg(pl.col("p").top_k(2).alias("_tk"))
    x = x.join(top2, on="cand_id").with_columns(
        cand_best_other=pl.when(pl.col("p") >= pl.col("_tk").list.get(0))
        .then(pl.col("_tk").list.get(1, null_on_oob=True)).otherwise(pl.col("_tk").list.get(0)).fill_null(0.0),
    ).drop("_tk")
    x = x.with_columns(
        s1_sum_p=pl.col("p").sum().over("s1_id"),
        s1_max_p=pl.col("p").max().over("s1_id"),
        s1_n_p50=(pl.col("p") > 0.5).sum().over("s1_id"),
        s1_n_conf=(pl.col("p") > CONF).sum().over("s1_id"),
        p_gap_up=(pl.col("p").shift(1).over("s1_id", order_by="p_rank") - pl.col("p")).fill_null(0.0),
        is_conf=(pl.col("p") > CONF).cast(pl.Int32),
    )
    conf = x.filter(pl.col("is_conf") == 1)
    same_num = conf.filter(pl.col("t_num") != "").group_by("s1_id", "t_num").len("m_num")
    same_name = conf.filter(pl.col("t_name") != "").group_by("s1_id", "t_name").len("m_name")
    cons = (conf.filter(pl.col("t_num") != "").group_by("s1_id", "t_num").len("c")
            .sort(["s1_id", "c", "t_num"], descending=[False, True, False])
            .group_by("s1_id", maintain_order=True).first().select("s1_id", pl.col("t_num").alias("cons_num")))
    x = (x.join(same_num, on=["s1_id", "t_num"], how="left").join(same_name, on=["s1_id", "t_name"], how="left")
         .join(cons, on="s1_id", how="left"))
    # counts exclude the candidate itself when it is a confident member
    return x.with_columns(
        member_same_num=(pl.col("m_num").fill_null(0) - pl.when(pl.col("t_num") != "").then(pl.col("is_conf")).otherwise(0)),
        member_same_name=(pl.col("m_name").fill_null(0) - pl.when(pl.col("t_name") != "").then(pl.col("is_conf")).otherwise(0)),
        members_other=(pl.col("s1_n_conf") - pl.col("is_conf")),
        num_eq_cons=((pl.col("t_num") != "") & (pl.col("t_num") == pl.col("cons_num"))).cast(pl.Int8),
        num_ne_cons=((pl.col("t_num") != "") & pl.col("cons_num").is_not_null()
                     & (pl.col("t_num") != pl.col("cons_num"))).cast(pl.Int8),
        cons_eq_s1=((pl.col("q_num") != "") & (pl.col("q_num") == pl.col("cons_num"))).cast(pl.Int8),
    ).select(ID_COLS + ["p", "p_rank", "cand_best_other", "s1_sum_p", "s1_max_p", "s1_n_p50", "s1_n_conf",
                        "p_gap_up", "member_same_num", "member_same_name", "members_other",
                        "num_eq_cons", "num_ne_cons", "cons_eq_s1"]).rename({"p": "p_s1"})


def build_shards(split: str, stage1: str) -> None:
    """Write feats_s2/<split>_<country>__<stage1>.parquet for every country."""
    src = OOF_DIR / (f"{stage1}.parquet" if split == "train" else f"test_{stage1}.parquet")
    pred = pl.read_parquet(src)
    S2_DIR.mkdir(parents=True, exist_ok=True)
    for country in sorted(pred["country"].unique().to_list()):
        t0 = time.time()
        q, t = load_country(split, country)
        ta = t.select(pl.col("entity_id").alias("cand_id"), pl.col("a_num").alias("t_num"), pl.col("n_core").alias("t_name"))
        qa = q.select(pl.col("entity_id").alias("s1_id"), pl.col("a_num").alias("q_num"))
        del q, t
        f = cluster_features(pred.filter(pl.col("country") == country), ta, qa)
        f.write_parquet(S2_DIR / f"{split}_{country}__{stage1}.parquet")
        print(f"[{split}/{country}] stage-2 rows={f.height:,} secs={time.time() - t0:.0f}", flush=True)


def load_s2(split: str, stage1: str, base: pl.DataFrame, country: str) -> pl.DataFrame:
    """Join stage-2 shard columns onto a base(+extra) feature frame of one country."""
    return base.join(pl.read_parquet(S2_DIR / f"{split}_{country}__{stage1}.parquet"), on=ID_COLS, how="left")


def train_stage2(stage1: str, name: str, folds: int = 3, lr: float = 0.08) -> None:
    """GPU XGBoost on base + extra + stage-2 features, same S1-hash folds, OOF report."""
    df = load_train_features(use_extra=True)
    df = pl.concat([load_s2("train", stage1, df.filter(pl.col("country") == c), c)
                    for c in sorted(df["country"].unique().to_list())])
    df = df.with_columns((pl.col("s1_id").hash(seed=SEED) % folds).cast(pl.UInt8).alias("fold"))
    feats = [c for c in df.columns if c not in ID_COLS + ["y", "fold", "country"]]
    print(f"rows={df.height:,} n_feats={len(feats)}", flush=True)
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy()
    fold = df["fold"].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    params = {**XGB_PARAMS, "eta": lr, "device": "cuda"}
    for k in range(folds):
        t0 = time.time()
        tr, va = fold != k, fold == k
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], feature_names=feats, max_bin=XGB_PARAMS["max_bin"])
        dva = xgb.QuantileDMatrix(X[va], y[va], feature_names=feats, ref=dtr)
        b = xgb.train(params, dtr, 3000, evals=[(dva, "valid")], early_stopping_rounds=100, verbose_eval=False)
        oof[va] = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        b.save_model(str(MODEL_DIR / f"{name}_fold{k}.json"))
        print(f"  fold {k}: best_iter={b.best_iteration} logloss={b.best_score:.5f} secs={time.time() - t0:.0f}", flush=True)
        del dtr, dva
    out = df.select(ID_COLS + ["country", "y"]).with_columns(pl.Series("p", oof))
    out.write_parquet(OOF_DIR / f"{name}.parquet")
    for k, v in evaluate(out, out["country"].unique().to_list()).items():
        print(k, v, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="xgb_v4x")
    ap.add_argument("--name", default="s2_v5")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="only build test shards (needs test_<stage1> preds)")
    a = ap.parse_args()
    if a.test_only:
        build_shards("test", a.stage1)
    else:
        if not a.skip_build:
            build_shards("train", a.stage1)
        train_stage2(a.stage1, a.name)
