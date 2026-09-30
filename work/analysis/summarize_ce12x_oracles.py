"""Conditional train-label oracles for prioritizing candidate-recovery routes."""
from pathlib import Path
import json
import polars as pl

OUT=Path(__file__).resolve().parent/'ce12x_review'
result={}
for c in ('India','US'):
    s=pl.read_parquet(OUT/f'{c}_ce12x_entity_scores.parquet')
    e=pl.read_parquet(OUT/f'{c}_information_residuals.parquet')
    slices={
        'blocking_nonempty_address':(pl.col('error')=='BLOCKING_FN')&~pl.col('empty'),
        'blocking_empty_address':(pl.col('error')=='BLOCKING_FN')&pl.col('empty'),
        'blocking_with_exact_predicted_anchor_name':(pl.col('error')=='BLOCKING_FN')&pl.col('predicted_anchor_exact_name'),
        'retrieved_FN_with_exact_predicted_anchor_name':(pl.col('error')=='FN')&pl.col('predicted_anchor_exact_name'),
        'blocking_low_name_high_addr':(pl.col('error')=='BLOCKING_FN')&(pl.col('raw_name_tset')<40)&(pl.col('raw_addr_tset')>=85)&~pl.col('empty'),
    }
    r={'n_S1':s.height,'baseline':s['f05'].mean(),'oracles':{}}
    for label,expr in slices.items():
        subset=e.filter(expr)
        d=s.join(subset.group_by('s1_id').len('added'),on='s1_id',how='left').with_columns(pl.col('added').fill_null(0))
        f=d.select(pl.when((pl.col('n_true')==0)&(pl.col('n_pred')==0)).then(1.).otherwise(1.25*(pl.col('tp')+pl.col('added'))/(.25*pl.col('n_true')+pl.col('n_pred')+pl.col('added'))).mean().alias('score')).item()
        r['oracles'][label]={'pairs':subset.height,'score':f,'gain':f-r['baseline']}
    result[c]=r
print(json.dumps(result),flush=True)
(OUT/'targeted_oracles.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
