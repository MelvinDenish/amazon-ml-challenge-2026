"""Current-incumbent audit; never modifies production predictions or submissions."""
from pathlib import Path
import gc
import json
import sys
import time
import numpy as np
import polars as pl

WORK=Path(__file__).resolve().parents[1]
ART=WORK/'artifacts'
OUT=WORK/'analysis/ce12x_review'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs
from postprocess import exclusive_owner_prob,one_owner,select_expected_f05
from score import per_entity_f05

KEY=['s1_id','cand_id']
MODELS={'ce12x':'test_blend_v7_ce_v12x_evaloof.parquet','ce12x3':'test_blend_v7_ce_v12x3_evaloof.parquet'}

def main():
    gt=load_ground_truth_pairs()
    result={'models':MODELS,'countries':{},'limitations':'Historical S1-held-out evaluation; global learned preprocessing, reused development set, incomplete owner competitors, no France truth. Oracles are diagnostic only.'}
    for c in ('India','US'):
        start=time.time()
        base=(pl.scan_parquet(ART/'oof/blend_v7.parquet').filter((pl.col('country')==c)&((pl.col('s1_id').hash(seed=45)%10000)<5000)).select(*KEY,pl.col('p').alias('base_p')).collect())
        ids=base['s1_id'].unique()
        truth=gt.join(pl.DataFrame({'s1_id':ids}),on='s1_id',how='semi')
        groups=(pl.scan_parquet(ART/f'groups/train_{c}.parquet').select(*KEY,'t_a_empty','twin_flag','form_disjoint','num_absdiff','n_tset','a_tset').join(base.lazy().select(KEY),on=KEY,how='semi').collect())
        cr={'n_s1':len(ids),'models':{}}
        frames={}
        for tag,file in MODELS.items():
            d=pl.scan_parquet(ART/'oof'/file).filter(pl.col('country')==c).collect()
            assert d.select(KEY).unique().height==d.height
            assert d.select(KEY).join(base.select(KEY),on=KEY,how='anti').height==0
            owned=one_owner(exclusive_owner_prob(d,'p'),'p')
            pred=select_expected_f05(owned,'p').select(KEY)
            found=d.filter(pl.col('y')==1).select(KEY)
            fp=pred.join(truth,on=KEY,how='anti')
            fn=found.join(pred,on=KEY,how='anti')
            missing=truth.join(found,on=KEY,how='anti')
            scores=per_entity_f05(truth,pred,ids)
            frames[tag]=scores.select('s1_id',pl.col('f05').alias(tag))
            scores.write_parquet(OUT/f'{c}_{tag}_entity_scores.parquet')
            def metric(p): return float(per_entity_f05(truth,p,ids)['f05'].mean())
            e=pl.concat([fp.with_columns(pl.lit('FP').alias('error')),fn.with_columns(pl.lit('FN').alias('error'))]).join(base,on=KEY).join(d.select(*KEY,pl.col('p').alias('stack_p')),on=KEY).join(groups,on=KEY,how='left').with_columns(
                ((pl.col('base_p')>.005)&(pl.col('base_p')<.995)).alias('routed'),
                pl.when(pl.col('t_a_empty')==1).then(pl.lit('empty_address')).when(pl.col('form_disjoint')==1).then(pl.lit('form_conflict')).when((pl.col('twin_flag')==1)&(pl.col('num_absdiff')<=10)).then(pl.lit('near_twin')).when(pl.col('twin_flag')==1).then(pl.lit('other_twin')).otherwise(pl.lit('rest')).alias('group'))
            e=e.join(owned.select(KEY).with_columns(pl.lit(True).alias('owns_candidate')),on=KEY,how='left').with_columns(pl.col('owns_candidate').fill_null(False))
            e.write_parquet(OUT/f'{c}_{tag}_pair_errors.parquet')
            missing.write_parquet(OUT/f'{c}_{tag}_blocking_misses.parquet')
            outside=e.filter(~pl.col('routed'))
            empty=e.filter((pl.col('error')=='FN')&(pl.col('group')=='empty_address')).select(KEY)
            diag={'macro_f05':float(scores['f05'].mean()),'candidate_oracle':metric(found),
                  'false_positives':fp.height,'retrieved_false_negatives':fn.height,'blocking_misses':missing.height,
                  'remove_FP_oracle':metric(pred.join(truth,on=KEY,how='semi')),
                  'recover_retrieved_FN_oracle':metric(pl.concat([pred,fn])),
                  'recover_blocking_misses_oracle':metric(pl.concat([pred,missing])),
                  'recover_empty_address_FN_oracle':metric(pl.concat([pred,empty])),
                  'fix_nonrouted_oracle':metric(pl.concat([pred.join(outside.filter(pl.col('error')=='FP').select(KEY),on=KEY,how='anti'),outside.filter(pl.col('error')=='FN').select(KEY)])),
                  'error_groups':e.group_by('group','error').len().sort('group','error').to_dicts(),
                  'routing_ownership':e.group_by('error','routed','owns_candidate').len().sort('error','routed','owns_candidate').to_dicts()}
            no_empty=fn.join(empty,on=KEY,how='anti')
            diag['recover_nonempty_address_FN_oracle']=metric(pl.concat([pred,no_empty]))
            ent=scores.with_columns(pl.when(pl.col('f05')==1).then(pl.lit('perfect')).when(pl.col('n_true')==0).then(pl.lit('singleton_FP')).when((pl.col('n_pred')>pl.col('tp'))&(pl.col('n_true')>pl.col('tp'))).then(pl.lit('both')).when(pl.col('n_pred')>pl.col('tp')).then(pl.lit('FP_only')).otherwise(pl.lit('FN_only')).alias('error'))
            diag['entity_errors']=ent.group_by('error').agg(pl.len().alias('n_s1'),((1-pl.col('f05')).sum()/len(ids)).alias('macro_loss')).to_dicts()
            cr['models'][tag]=diag
            print(json.dumps({'country':c,'model':tag,**diag}),flush=True)
            del d,owned,pred,found,fp,fn,e
            gc.collect()
        paired=frames['ce12x'].join(frames['ce12x3'],on='s1_id').with_columns((pl.col('ce12x3')-pl.col('ce12x')).alias('delta'))
        delta=paired['delta'].to_numpy()
        cr['synthetic_increment']={'mean_delta':float(delta.mean()),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'paired_SE_assuming_independent_S1':float(delta.std(ddof=1)/np.sqrt(len(delta)))}
        result['countries'][c]=cr
        (OUT/'audit_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'country':c,'comparison':cr['synthetic_increment'],'seconds':time.time()-start}),flush=True)
        del base,groups,truth,frames
        gc.collect()

if __name__=='__main__':main()
