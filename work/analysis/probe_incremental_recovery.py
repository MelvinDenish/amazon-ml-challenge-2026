"""Offline pilot: narrowly retrieve new pairs without rebuilding existing candidates."""
from pathlib import Path
import sys,json,gc
import polars as pl
from rapidfuzz import process,fuzz
W=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(W/'code/business_entity_resolution/src'))
from io_utils import load_records,load_ground_truth_pairs
from postprocess import exclusive_owner_prob,one_owner,select_expected_f05
from score import per_entity_f05
from swap_rules import TOK,NUM,ADDR
O=W/'analysis/final_slot_review'
K=['s1_id','cand_id']

def main():
    gt=load_ground_truth_pairs()
    results={}
    for split,c in [('train','US'),('train','India'),('test','France')]:
        file='test_v10_ce8_evaloof.parquet' if split=='train' else 'test_v11_noun.parquet'
        pred=pl.scan_parquet(W/'artifacts/oof'/file).filter(pl.col('country')==c).collect()
        ids=pred['s1_id'].unique()
        q=load_records(split,'source1',c).select(pl.col('entity_id').alias('s1_id'),TOK.list.join(' ').alias('name_key'),NUM.alias('num'),ADDR.alias('qa'))
        # Count every reference, including those outside the evaluation queries.
        q=q.with_columns(nq=pl.len().over('name_key','num')).filter((pl.col('nq')==1)&pl.col('num').is_not_null()&(pl.col('name_key').str.len_chars()>=6))
        q=q.join(pl.DataFrame({'s1_id':ids}),on='s1_id',how='semi')
        t=pl.concat([load_records(split,s,c) for s in ('source2','source3')]).select(pl.col('entity_id').alias('cand_id'),TOK.list.join(' ').alias('name_key'),NUM.alias('num'),ADDR.alias('ta'))
        ob=q.join(t,on=['name_key','num']).select(*K,'qa','ta')
        del q,t
        sim=process.cpdist(ob['qa'].to_list(),ob['ta'].to_list(),scorer=fuzz.ratio,workers=-1)
        ob=ob.with_columns(sim=pl.Series(sim)).filter((pl.col('sim')>=95)&(pl.col('qa').str.len_chars()>=12)&(pl.col('ta').str.len_chars()>=12))
        # This pilot deliberately considers only missing candidates, not threshold changes.
        ob=ob.join(pred.select(K),on=K,how='anti')
        sel=select_expected_f05(one_owner(exclusive_owner_prob(pred.select(*K,'p'),'p'),'p'),'p').select(K)
        ob=ob.join(sel.select('cand_id'),on='cand_id',how='anti')
        result={'n_s1':len(ids),'new_unowned_pairs':ob.height,'affected_s1':ob['s1_id'].n_unique()}
        if split=='train':
            truth=gt.join(pl.DataFrame({'s1_id':ids}),on='s1_id',how='semi')
            y=ob.join(gt.with_columns(y=pl.lit(1)),on=K,how='left').with_columns(pl.col('y').fill_null(0))
            a=per_entity_f05(truth,sel,ids)
            b=per_entity_f05(truth,pl.concat([sel,ob.select(K)]),ids)
            result.update(true_pairs=int(y['y'].sum()),precision=float(y['y'].mean()) if y.height else None,
                base_f05=float(a['f05'].mean()),new_f05=float(b['f05'].mean()),delta=float(b['f05'].mean()-a['f05'].mean()))
        else:
            sizes=sel.group_by('s1_id').len('n_old')
            result['old_set_sizes']=ob.join(sizes,on='s1_id',how='left').with_columns(pl.col('n_old').fill_null(0)).group_by('n_old').len().sort('n_old').to_dicts()
        ob.write_parquet(O/(split+'_'+c+'_incremental_retrieval.parquet'))
        results[split+'_'+c]=result
        (O/'incremental_recovery_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps({split+'_'+c:result}),flush=True)
        del pred,sel,ob
        gc.collect()

if __name__=='__main__':main()
