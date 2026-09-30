"""Read-only cohort and submission-difference audit for the 0.985 plan."""
from pathlib import Path
import json
import sys
import polars as pl

WORK = Path(__file__).resolve().parents[1]
ART = WORK / 'artifacts'
sys.path.insert(0, str(WORK / 'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs

out = {}
ev = pl.read_parquet(WORK / 'kaggle_jobs/bedrock_ce/ce_eval.parquet', columns=['s1_id','cand_id'])
tr2 = pl.read_parquet(WORK / 'kaggle_jobs/bedrock_ce/ce_train2.parquet', columns=['s1_id','cand_id','y'])
out['eval_train_S1_overlap'] = ev.select('s1_id').unique().join(tr2.select('s1_id').unique(), on='s1_id').height
scores = pl.concat([pl.read_parquet(WORK / f'analysis/{c}_ce1s_entity_scores.parquet').select('s1_id',pl.col('f05').alias('ce1s')).join(
    pl.read_parquet(WORK / f'analysis/{c}_ce12_entity_scores.parquet').select('s1_id',pl.col('f05').alias('ce12')),on='s1_id'
    ).with_columns(pl.lit(c).alias('country')) for c in ('India','US')])
seen = ev.join(tr2.select('cand_id').unique(),on='cand_id',how='semi').select('s1_id').unique().with_columns(pl.lit(True).alias('target_seen'))
scores = scores.join(seen,on='s1_id',how='left').with_columns(pl.col('target_seen').fill_null(False))
out['cohorts'] = scores.group_by('country','target_seen').agg(pl.len().alias('S1s'),pl.col('ce1s','ce12').mean(),
    (pl.col('ce12')-pl.col('ce1s')).mean().alias('delta')).sort('country','target_seen').to_dicts()
out['overall'] = scores.select(pl.col('ce1s','ce12').mean(),(pl.col('ce12')-pl.col('ce1s')).mean().alias('delta')).to_dicts()[0]
out['v2_target_reuse_by_training_label'] = ev.join(tr2.group_by('cand_id').agg(pl.col('y').max().alias('seen_positive')),on='cand_id',how='inner').group_by('seen_positive').len().to_dicts()

old = pl.read_csv(WORK/'submissions/v07_ce1s_own/matching_results.tsv',separator='\t',quote_char=None,missing_utf8_is_empty_string=True).rename({'matched_entity_ids':'old'})
new = pl.read_csv(WORK/'submissions/v07_ce12_own/matching_results.tsv',separator='\t',quote_char=None,missing_utf8_is_empty_string=True).rename({'matched_entity_ids':'new'})
changes = old.join(new,on='source1_entity_id').filter(pl.col('old') != pl.col('new')).rename({'source1_entity_id':'s1_id'})
s1 = pl.read_parquet(ART/'data/test_source1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
changes = changes.join(s1,on='s1_id')
out['test_changed_S1s'] = changes.height
out['test_changed_by_country'] = changes.group_by('country').agg(pl.len().alias('changed_S1'),
    (pl.col('old')=='').sum().alias('empty_to_nonempty'),(pl.col('new')=='').sum().alias('nonempty_to_empty')).to_dicts()
for tag in ('old','new'):
    pairs = changes.filter(pl.col(tag)!='').select('s1_id','country',pl.col(tag).str.split(',').alias('cand_id')).explode('cand_id')
    if tag=='old': oldpairs=pairs
    else: newpairs=pairs
out['test_removed_pairs'] = oldpairs.join(newpairs.select('s1_id','cand_id'),on=['s1_id','cand_id'],how='anti').group_by('country').len().to_dicts()
out['test_added_pairs'] = newpairs.join(oldpairs.select('s1_id','cand_id'),on=['s1_id','cand_id'],how='anti').group_by('country').len().to_dicts()
out['test_country_sizes'] = s1.group_by('country').len().to_dicts()
(WORK/'analysis/ce_cohort_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out),flush=True)
