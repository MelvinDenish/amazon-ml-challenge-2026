"""Bounded CPU-only ablation of CE stacker other-max semantics.

Writes analysis results only. Uses historical two-fold evaluation, not a fresh holdout.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import polars as pl
import xgboost as xgb

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'code/business_entity_resolution/src'))
from ce_stack import features, feat_names, PARAMS
from io_utils import load_ground_truth_pairs
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05
from score import per_entity_f05

ART = WORK / 'artifacts'
KEY = ['s1_id', 'cand_id']
base = (pl.scan_parquet(ART / 'oof/blend_v7.parquet')
        .filter((pl.col('s1_id').hash(seed=45) % 10000) < 5000).collect())
ce = pl.read_parquet(ART / 'ce/ce_eval_scores.parquet').rename({'ce': 'ce1'})
d = base.join(ce, on=KEY, how='left')
r = features(d, ['ce1'], 'train')
fn = feat_names(['ce1'])
y = r['y'].to_numpy()
fold = (r['s1_id'].hash(seed=99) % 2).to_numpy()
truth = load_ground_truth_pairs().join(base.select('s1_id').unique(), on='s1_id', how='semi')
results = {}
for mode in ('original', 'correct_other_max'):
    start = time.time()
    frame = r
    if mode == 'correct_other_max':
        for group, column in [('s1_id', 'ce_other_max'), ('cand_id', 'ce_cand_other_max')]:
            frame = frame.with_columns(pl.when(pl.col('ce_mean') == pl.col('ce_mean').max().over(group))
                .then(pl.col(column)).otherwise(pl.col('ce_mean').max().over(group)).alias(column))
    X = frame.select(fn).to_numpy().astype(np.float32)
    probs = np.zeros(len(y), dtype=np.float32)
    iters = []
    for k in (0, 1):
        tr = xgb.DMatrix(X[fold != k], label=y[fold != k], feature_names=fn)
        va = xgb.DMatrix(X[fold == k], label=y[fold == k], feature_names=fn)
        model = xgb.train({**PARAMS, 'device': 'cpu', 'nthread': 4}, tr, 1200,
            evals=[(va, 'valid')], early_stopping_rounds=100, verbose_eval=False)
        probs[fold == k] = model.predict(va, iteration_range=(0, model.best_iteration + 1))
        iters.append(model.best_iteration + 1)
    stacked = (base.join(r.select(KEY).with_columns(pl.Series('q', probs)), on=KEY, how='left')
               .with_columns(pl.coalesce('q', 'p').alias('p')).drop('q'))
    scores = []
    for c in ('India', 'US'):
        o = stacked.filter(pl.col('country') == c)
        selected = select_expected_f05(one_owner(exclusive_owner_prob(o, 'p'), 'p'), 'p')
        scores.append(per_entity_f05(truth, selected, o['s1_id'].unique()).with_columns(pl.lit(c).alias('country')))
    scores = pl.concat(scores)
    scores.write_parquet(WORK / f'analysis/ce_context_{mode}_entity_scores.parquet')
    results[mode] = {'countries': scores.group_by('country').agg(pl.col('f05').mean()).to_dicts(),
                     'pooled': scores['f05'].mean(), 'best_iterations': iters, 'seconds': time.time()-start}
    print(json.dumps({mode: results[mode]}), flush=True)
(WORK / 'analysis/ce_context_probe_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
