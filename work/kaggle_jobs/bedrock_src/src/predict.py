"""Score test candidates with the trained fold models and write a submission.

    python predict.py --name lgb_v1 --version v02_lgb_v1 [--select ef05|thr --thr 0.5]

Fold models are averaged. Post-processing: global one-owner on p, then
expected-F0.5 set selection (default) or a plain probability threshold.
"""

import argparse

import numpy as np
import polars as pl
import xgboost as xgb

from config import DEVICE
from features import FEAT_DIR, ID_COLS
from io_utils import load_records
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05, select_threshold
from submission import write_submission
from train import MODEL_DIR, OOF_DIR


def load_models(name: str) -> tuple[list, list[str], str]:
    """Load fold models of a run: LightGBM (.txt) or XGBoost (.json).

    Returns (models, feature_names, kind).
    """
    lgb_files = sorted(MODEL_DIR.glob(f"{name}_fold*.txt"))
    if lgb_files:
        import lightgbm as lgb  # lazy: only LightGBM runs need it
        models = [lgb.Booster(model_file=str(f)) for f in lgb_files]
        return models, models[0].feature_name(), "lgb"
    cat_files = sorted(MODEL_DIR.glob(f"{name}_fold*.cbm"))
    if cat_files:
        import json

        from catboost import CatBoostClassifier  # lazy: only CatBoost runs need it
        models = []
        for f in cat_files:
            m = CatBoostClassifier()
            m.load_model(str(f))
            models.append(m)
        feats = json.load(open(MODEL_DIR / f"{name}_features.json"))
        return models, feats, "cat"
    models = []
    for f in sorted(MODEL_DIR.glob(f"{name}_fold*.json")):
        b = xgb.Booster()
        b.load_model(str(f))
        b.set_param({"device": DEVICE})
        models.append(b)
    if not models:
        raise FileNotFoundError(f"no fold models named {name}_fold* in {MODEL_DIR}")
    return models, models[0].feature_names, "xgb"


def predict_matrix(models: list, kind: str, X: np.ndarray, feats: list[str]) -> np.ndarray:
    """Average the fold models' probabilities for a feature matrix."""
    if kind == "lgb":
        return np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
    if kind == "cat":
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)
    # Row chunks keep GPU memory bounded (21M rows x 66 features needs >10 GB at once).
    out = np.empty(len(X), dtype=np.float32)
    step = 2_000_000
    for s in range(0, len(X), step):
        d = xgb.DMatrix(X[s:s + step], feature_names=feats)
        out[s:s + step] = np.mean([m.predict(d) for m in models], axis=0)
    return out


def predict_test(name: str, stage1: str | None = None) -> pl.DataFrame:
    """Return test pairs with the fold-averaged probability ``p``.

    Extra-feature shards are joined when the model needs them; for a stage-2
    model (``stage1`` given) the stage-2 shards built from that stage-1 model's
    test predictions are joined too.
    """
    models, feats, kind = load_models(name)
    parts = []
    for f in sorted(FEAT_DIR.glob("test_*.parquet")):
        df = pl.read_parquet(f)
        if any(c not in df.columns for c in feats):  # model uses extra_features.py columns
            df = df.join(pl.read_parquet(FEAT_DIR.parent / "feats_extra" / f.name), on=ID_COLS, how="left")
        if stage1 and any(c not in df.columns for c in feats):  # stage-2 model
            s2 = FEAT_DIR.parent / "feats_s2" / f"{f.stem}__{stage1}.parquet"
            df = df.join(pl.read_parquet(s2), on=ID_COLS, how="left")
        X = df.select(feats).to_numpy().astype(np.float32)
        p = predict_matrix(models, kind, X, feats)
        parts.append(df.select(ID_COLS).with_columns(pl.Series("p", p), pl.lit(f.stem.split("_", 1)[1]).alias("country")))
    return pl.concat(parts)


def predict_blend(spec_name: str) -> pl.DataFrame:
    """Weighted average of several models' test probabilities (weights from ensemble.py)."""
    import json
    spec = json.load(open(MODEL_DIR / f"{spec_name}.json"))
    out = None
    for n, w in zip(spec["models"], spec["weights"]):
        if w == 0:
            continue
        p = predict_test(n).rename({"p": n})
        p.write_parquet(OOF_DIR / f"test_{n}.parquet")
        out = p if out is None else out.join(p.select(ID_COLS + [n]), on=ID_COLS)
    active = [(n, w) for n, w in zip(spec["models"], spec["weights"]) if w > 0]
    blend_p = sum(pl.col(n) * w for n, w in active)
    return out.select(ID_COLS + ["country", blend_p.cast(pl.Float32).alias("p")])


def main() -> None:
    """Predict, post-process, write + validate + log the submission."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgb_v1")
    ap.add_argument("--version", required=True)
    ap.add_argument("--select", default="ef05", choices=["ef05", "thr"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--stage1", default=None, help="stage-1 model name (for stage-2 models)")
    ap.add_argument("--blend", action="store_true", help="--name is a blend spec from ensemble.py")
    ap.add_argument("--owner-prob", action="store_true",
                    help="exclusive-owner probabilities before one-owner/selection (postprocess.exclusive_owner_prob)")
    ap.add_argument("--reuse", action="store_true", help="re-select from the saved oof/test_<name>.parquet")
    a = ap.parse_args()
    if a.reuse:
        pred = pl.read_parquet(OOF_DIR / f"test_{a.name}.parquet")
    else:
        pred = predict_blend(a.name) if a.blend else predict_test(a.name, a.stage1)
        pred.write_parquet(OOF_DIR / f"test_{a.name}.parquet")
    owned = one_owner(exclusive_owner_prob(pred, "p") if a.owner_prob else pred, "p")
    matches = select_expected_f05(owned, "p") if a.select == "ef05" else select_threshold(owned, "p", a.thr)
    # Label-free sanity check against train (5.6% singletons, 3.46 matches per S1).
    s1 = load_records("test", "source1").select(pl.col("entity_id").alias("s1_id"), "country")
    per_s1 = s1.join(matches.group_by("s1_id").len("n"), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
    stats = per_s1.group_by("country").agg(
        pl.len().alias("n_s1"), (pl.col("n") == 0).mean().alias("empty_rate"), pl.col("n").mean().alias("mean_matches"))
    print("test sanity by country:", stats.sort("country").to_dicts())
    write_submission(a.version, matches, pred, notes=f"{a.name} select={a.select} thr={a.thr} owner_prob={a.owner_prob}")


if __name__ == "__main__":
    main()
