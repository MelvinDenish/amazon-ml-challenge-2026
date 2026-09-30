"""Conservative exact-name anchor pilot on saved OOF. Diagnostic, not honest new CV.

Anchors use OOF probabilities, but competition is the historical sampled S1
population and preprocessing is the historical globally learned map. No test
predictions or production files are changed.
"""
from pathlib import Path
import gc
import json
import sys
import polars as pl

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'code/business_entity_resolution/src'))
from normalize import normalize_names, normalize_addresses, transliterate
from postprocess import one_owner, select_expected_f05
from io_utils import load_ground_truth_pairs
from score import per_entity_f05

ART=WORK/'artifacts'
OUT=WORK/'analysis'
V6=ART/'k2_out_v2/artifacts'


def main():
    results={}
    truth=load_ground_truth_pairs()
    mapping=pl.read_parquet(ART/'translit_map.parquet')
    for country in ('India','US'):
        p=pl.scan_parquet(V6/'oof/xgb_v6.parquet').filter(pl.col('country')==country).collect()
        owned=one_owner(p,'p')
        selected=select_expected_f05(owned)
        needed=p.select('cand_id').unique()
        raw=pl.concat([pl.scan_parquet(ART/f'data/train_{s}.parquet').filter(pl.col('country')==country).rename({'entity_id':'cand_id'}) for s in ('source2','source3')])
        raw=raw.join(needed.lazy(),on='cand_id',how='semi').collect()
        attrs=transliterate(normalize_addresses(normalize_names(raw)),mapping).select('cand_id','n_full','a_empty')
        del raw
        anchors=selected.filter(pl.col('p')>=0.999).join(attrs,on='cand_id').filter(~pl.col('a_empty'))
        anchors=anchors.group_by('n_full').agg(pl.col('s1_id').n_unique().alias('n_owners'),pl.col('s1_id').first(),pl.len().alias('support'),pl.col('cand_id').str.slice(0,2).n_unique().alias('n_sources'))
        anchors=anchors.filter((pl.col('n_owners')==1)&(pl.col('support')>=2)&(pl.col('n_sources')==2))
        unmatched=attrs.filter(pl.col('a_empty')).join(selected.select('cand_id'),on='cand_id',how='anti')
        proposals=unmatched.join(anchors,on='n_full').select('s1_id','cand_id','support')
        # Use only already-retrieved edges. Ground truth is attached for evaluation only.
        proposals=proposals.join(p.select('s1_id','cand_id','y','p'),on=['s1_id','cand_id'])
        proposals.write_parquet(OUT/f'{country}_bridge_proposals.parquet')
        combined=pl.concat([selected.select('s1_id','cand_id'),proposals.select('s1_id','cand_id')]).unique()
        ids=p['s1_id'].unique()
        baseline=per_entity_f05(truth,selected,ids)
        updated=per_entity_f05(truth,combined,ids)
        result={'proposed_pairs':proposals.height,'proposal_precision':proposals['y'].mean(),
                'baseline_f05':baseline['f05'].mean(),'pilot_f05':updated['f05'].mean(),
                'delta':updated['f05'].mean()-baseline['f05'].mean(),
                'limitations':'Same historical OOF population and global preprocessing; no independent tuning split; not a leaderboard gain or production-ready rule.'}
        results[country]=result
        print(json.dumps({country:result}),flush=True)
        (OUT/'bridge_probe_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        del p,owned,selected,needed,attrs,anchors,unmatched,proposals,combined,baseline,updated
        gc.collect()


if __name__=='__main__':
    main()
