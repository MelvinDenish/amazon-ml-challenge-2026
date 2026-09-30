"""Diagnostic: observable ambiguity and recovery signals in current residuals.

Uses labels only for train diagnostics; never creates test labels or predictions.
"""
from pathlib import Path
import gc
import json
import sys
import polars as pl
from rapidfuzz import fuzz,process

WORK=Path(__file__).resolve().parents[1]
ART=WORK/'artifacts'
OUT=WORK/'analysis/ce12x_review'
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs

EMPTY=pl.col('business_address').fill_null('').str.strip_chars().str.to_lowercase().is_in(['','null','<null>','none','n/a'])
FOLD=pl.col('business_name').str.normalize('NFKD').str.replace_all(r'\p{M}','').str.to_lowercase().str.replace_all(r'[^\p{L}\p{N}]','')
ASCII_FOLD=pl.col('business_name').str.normalize('NFKD').str.replace_all(r'\p{M}','').str.to_lowercase().str.replace_all(r'[^a-z0-9]','')
KEY=['s1_id','cand_id']

def main():
    truth=load_ground_truth_pairs()
    owners=truth.rename({'s1_id':'true_owner'})
    result={}
    for c in ('India','US'):
        errs=pl.read_parquet(OUT/f'{c}_ce12x_pair_errors.parquet')
        misses=pl.read_parquet(OUT/f'{c}_ce12x_blocking_misses.parquet')
        d=pl.concat([errs.select(*KEY,'error'),misses.with_columns(pl.lit('BLOCKING_FN').alias('error'))])
        targets=pl.concat([pl.scan_parquet(ART/f'data/train_{s}.parquet').filter(pl.col('country')==c).select('entity_id','business_name','business_address') for s in ('source2','source3')]).with_columns(
            EMPTY.alias('empty'),FOLD.alias('fold'),ASCII_FOLD.alias('ascii_fold'),pl.col('entity_id').str.slice(0,2).alias('source')).collect()
        empt=targets.filter(pl.col('empty')).join(owners,left_on='entity_id',right_on='cand_id',how='left').with_columns(pl.col('true_owner').fill_null('NO_OWNER'))
        exact=empt.group_by('source','business_name','business_address').agg(pl.len().alias('identical_records'),pl.col('true_owner').n_unique().alias('identical_owner_states'))
        folded=empt.group_by('source','fold').agg(pl.len().alias('folded_records'),pl.col('true_owner').n_unique().alias('folded_owner_states'))
        annot=empt.join(exact,on=['source','business_name','business_address']).join(folded,on=['source','fold']).select('entity_id','identical_records','identical_owner_states','folded_records','folded_owner_states')
        d=d.join(targets.rename({'entity_id':'cand_id','business_name':'t_name','business_address':'t_addr'}),on='cand_id').join(annot.rename({'entity_id':'cand_id'}),on='cand_id',how='left')
        q=pl.scan_parquet(ART/'data/train_source1.parquet').filter(pl.col('country')==c).select(pl.col('entity_id').alias('s1_id'),pl.col('business_name').alias('q_name'),pl.col('business_address').alias('q_addr')).collect()
        d=d.join(q,on='s1_id')
        nr=process.cpdist(d['q_name'].to_list(),d['t_name'].to_list(),scorer=fuzz.token_set_ratio,workers=4)
        ar=process.cpdist(d['q_addr'].to_list(),d['t_addr'].to_list(),scorer=fuzz.token_set_ratio,workers=4)
        d=d.with_columns(pl.Series('raw_name_tset',nr),pl.Series('raw_addr_tset',ar),pl.col('t_name').str.contains(r'[^\x00-\x7F]').alias('non_ascii_name'))
        # Anchor name availability: predicted high-confidence, addressed members only.
        ev=pl.scan_parquet(ART/'oof/test_blend_v7_ce_v12x_evaloof.parquet').filter(pl.col('country')==c)
        anchor=ev.filter(pl.col('p')>=.995).select(KEY).collect().join(targets.select(pl.col('entity_id').alias('cand_id'),'fold','empty'),on='cand_id').filter(~pl.col('empty')&(pl.col('fold')!=''))
        anchor=anchor.select('s1_id','fold').unique().with_columns(pl.lit(True).alias('predicted_anchor_exact_name'))
        d=d.join(anchor,on=['s1_id','fold'],how='left').with_columns(pl.col('predicted_anchor_exact_name').fill_null(False))
        info={
            'empty_target_population':empt.height,
            'ASCII_fold_erases_nonempty_name_count':empt.filter((pl.col('business_name').str.len_chars()>0)&(pl.col('ascii_fold')=='')).height,
            'error_slices':d.group_by('error').agg(pl.len().alias('pairs'),pl.col('empty').sum(),
                (pl.col('empty')&(pl.col('identical_owner_states')>1)).sum().alias('strict_identical_input_multiple_owner_states'),
                (pl.col('empty')&(pl.col('folded_owner_states')>1)).sum().alias('folded_name_multiple_owner_states'),
                pl.col('predicted_anchor_exact_name').sum().alias('anchor_exact_name'),
                ((pl.col('raw_name_tset')<40)&(pl.col('raw_addr_tset')>=85)&~pl.col('empty')).sum().alias('low_name_high_address_raw'),
                pl.col('non_ascii_name').sum()).to_dicts(),
            'empty_FN_ambiguity':d.filter((pl.col('error')=='FN')&pl.col('empty')).group_by(
                (pl.col('identical_owner_states')>1).alias('strict_ambiguous'),(pl.col('folded_owner_states')>1).alias('folded_ambiguous')).len().to_dicts()
        }
        d.write_parquet(OUT/f'{c}_information_residuals.parquet')
        d.sort('s1_id','cand_id').group_by('error',maintain_order=True).head(30).write_csv(OUT/f'{c}_residual_examples.tsv',separator='\t')
        result[c]=info
        print(json.dumps({'country':c,**info}),flush=True)
        (OUT/'information_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        del targets,empt,exact,folded,annot,d,q
        gc.collect()

if __name__=='__main__':main()
