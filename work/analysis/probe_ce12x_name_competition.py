"""Describe full-country reference-name competition for residual errors."""
from pathlib import Path
import json
import sys
import polars as pl

WORK=Path(__file__).resolve().parents[1]
ART=WORK/'artifacts'
OUT=WORK/'analysis/ce12x_review'
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from normalize import normalize_names

result={}
for country in ('India','US'):
    q=pl.scan_parquet(ART/'data/train_source1.parquet').filter(pl.col('country')==country).select('entity_id','business_name').collect()
    q=normalize_names(q).with_columns(pl.col('n_toks').list.sort().list.join(' ').alias('namekey')).select(pl.col('entity_id').alias('s1_id'),'n_full','namekey')
    q=q.with_columns(pl.len().over('namekey').alias('S1_core_name_multiplicity'),pl.len().over('n_full').alias('S1_full_name_multiplicity'))
    errors=pl.read_parquet(OUT/f'{country}_information_residuals.parquet').join(q,on='s1_id')
    result[country]=errors.group_by('error','empty').agg(pl.len().alias('pairs'),
        (pl.col('S1_core_name_multiplicity')>1).sum().alias('S1_shares_core_name_with_other_S1'),
        (pl.col('S1_full_name_multiplicity')>1).sum().alias('S1_shares_full_name_with_other_S1'),
        pl.col('S1_core_name_multiplicity').median().alias('median_core_name_multiplicity')).to_dicts()
print(json.dumps(result),flush=True)
(OUT/'name_competition_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
