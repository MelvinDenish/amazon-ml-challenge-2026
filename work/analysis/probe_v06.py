"""Small diagnostic probes, using existing v06 artifacts; does not retrain."""
from pathlib import Path
import json
import sys
import numpy as np
import polars as pl
import xgboost as xgb

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from normalize import normalize_records
from postprocess import one_owner, select_expected_f05
from score import per_entity_f05
from io_utils import load_ground_truth_pairs

ART=WORK/'artifacts'
V6=ART/'k2_out_v2/artifacts'
OUT=WORK/'analysis'


def main():
    result={}
    # Verify the actual saved models include post-best trees; reproduce native API behavior.
    country='India'
    selector=(pl.col('s1_id').hash(seed=742)%10000)<200
    f=pl.scan_parquet(V6/f'feats/train_{country}.parquet').filter(selector).collect()
    e=pl.scan_parquet(V6/f'feats_extra/train_{country}.parquet').filter(selector).collect()
    f=f.join(e,on=['s1_id','cand_id'])
    saved=pl.scan_parquet(V6/'oof/xgb_v6.parquet').filter((pl.col('country')==country)&selector).collect()
    f=f.join(saved,on=['s1_id','cand_id']).with_columns((pl.col('s1_id').hash(seed=42)%3).alias('fold'))
    p_all=np.zeros(f.height,dtype=np.float32)
    p_best=np.zeros(f.height,dtype=np.float32)
    model_info=[]
    for k in range(3):
        b=xgb.Booster()
        b.load_model(V6/f'models/xgb_v6_fold{k}.json')
        b.set_param({'device':'cpu','nthread':4})
        mask=f['fold'].to_numpy()==k
        d=xgb.DMatrix(f.filter(pl.col('fold')==k).select(b.feature_names).to_numpy().astype(np.float32),feature_names=b.feature_names)
        p_all[mask]=b.predict(d)
        p_best[mask]=b.predict(d,iteration_range=(0,b.best_iteration+1))
        model_info.append({'fold':k,'best_rounds':b.best_iteration+1,'saved_rounds':b.num_boosted_rounds()})
    truth=load_ground_truth_pairs()
    def calc(p):
        pairs=f.select('s1_id','cand_id').with_columns(pl.Series('p',p))
        return float(per_entity_f05(truth,select_expected_f05(one_owner(pairs,'p')),pairs['s1_id'].unique())['f05'].mean())
    result['prediction_probe']={
        'sample':'2% hash sample of v06 India OOF S1; competition only within sample, so diagnostic, not headline CV',
        'n_pairs':f.height,'n_s1':f['s1_id'].n_unique(),'models':model_info,
        'max_best_vs_saved_oof_abs_diff':float(np.max(np.abs(p_best-f['p'].to_numpy()))),
        'mean_all_vs_best_abs_diff':float(np.mean(np.abs(p_all-p_best))),
        'pairs_changed_over_001':int((np.abs(p_all-p_best)>.01).sum()),
        'best_iteration_sample_f05':calc(p_best),
        'all_rounds_sample_f05':calc(p_all),
    }
    # Stress examples directly taken from false-positive audit rows.
    names=['Surgical Physicians Inc.','Surgical Physicians Inc.','Services Kvr (india) Pvt Ltd','Services Kvr (india) Pvt Ltd','Vibes & Associates','Vibes & Associates']
    addresses=['4421 1A Amethyst Court, High Point, NC','4421 12A Amethyst Court, High Point, NC',
               'Flat No A-5/101, Clifton Apartment, Faridabad, Haryana','FLAT NO A-5/114, FARIDABAD, Haryana',
               '30-25-4/1, Vadlapudi, Andhra Pradesh','30-25-4/4, Vadlapudi, Andhra Pradesh']
    df=pl.DataFrame({'business_name':names,'business_address':addresses})
    result['number_parser_examples']=normalize_records(df).select('business_address','a_norm','a_num','a_street').to_dicts()
    # Characterize ambiguity among exactly identical raw target texts, using train labels only.
    empty_names=[]
    for country in ('India','US'):
        target=pl.concat([pl.scan_parquet(ART/f'data/train_{s}.parquet').filter(pl.col('country')==country) for s in ('source2','source3')])
        target=target.filter(pl.col('business_address').str.strip_chars().is_in(['','<NULL>','null','NULL'])).collect()
        norm=normalize_records(target).select(pl.col('entity_id').alias('cand_id'),'business_name','n_full','n_nospace')
        mapping=norm.join(truth,on='cand_id',how='left')
        by_name=mapping.group_by('n_full').agg(pl.len().alias('records'),pl.col('s1_id').drop_nulls().n_unique().alias('owners'),pl.col('s1_id').null_count().alias('distractors'))
        ambiguous=mapping.join(by_name,on='n_full').filter((pl.col('owners')>1)|((pl.col('owners')>0)&(pl.col('distractors')>0)))
        empty_names.append({'country':country,'empty_address_target_records':norm.height,
            'records_sharing_exact_normalized_name_with_multiple_owners_or_distractors':ambiguous.height,
            'fraction_ambiguous':ambiguous.height/norm.height,
            'examples':by_name.filter(pl.col('owners')>1).sort('records',descending=True).head(5).to_dicts()})
    result['empty_address_ambiguity']=empty_names
    (OUT/'probe_v06_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
