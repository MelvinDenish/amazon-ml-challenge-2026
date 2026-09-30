"""Count remaining selected France noun conflicts; no labels inferred or output changed."""
from pathlib import Path
from collections import Counter
import json
import sys
import polars as pl
from rapidfuzz import fuzz

W=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(W/'code/business_entity_resolution/src'))
from io_utils import load_records
from swap_rules import TOK,FILLERS,STOP
O=W/'analysis/final_slot_review'

def main():
    d=pl.read_parquet(O/'baseline_selected_details.parquet')
    q=load_records('test','source1','France').select(TOK.alias('tokens'))
    freq=Counter(w for row in q['tokens'].to_list() for w in row)
    qtok=d.select('s1_id','cand_id',pl.col('qname').alias('business_name')).with_columns(TOK.alias('qt')).drop('business_name')
    ttok=d.select('s1_id','cand_id',pl.col('tname').alias('business_name')).with_columns(TOK.alias('tt')).drop('business_name')
    d=d.join(qtok,on=['s1_id','cand_id']).join(ttok,on=['s1_id','cand_id']).with_columns(
        add=pl.col('tt').list.set_difference('qt'),miss=pl.col('qt').list.set_difference('tt'))
    selected_counts=d.group_by('s1_id').len('old_n')
    wide=d.filter(pl.col('same_num') & (pl.col('addr_sim')>=85) & (pl.col('add').list.len()>0) & (pl.col('miss').list.len()>0)
        &(pl.col('add').list.len()<=3)&(pl.col('miss').list.len()<=3))
    ignore=set(FILLERS['France']+STOP)
    rows=[]
    for row in wide.iter_rows(named=True):
        add,miss=row['add'],row['miss']
        hit=None
        for a in add:
            for m in miss:
                if min(len(a),len(m))<4 or a in ignore or m in ignore or min(freq[a],freq[m])<30:continue
                sim=fuzz.ratio(a,m)
                if len(add)==len(miss)==1:
                    hit='one_noun_below70' if sim<70 else 'one_noun_close_spelling'
                elif sim<70:
                    ar=' '.join(sorted(x for x in add if x!=a)); mr=' '.join(sorted(x for x in miss if x!=m))
                    if ar and mr and fuzz.ratio(ar,mr)>=85:
                        hit='noun_plus_other_typo'
                if hit:break
            if hit:break
        if hit:rows.append((row['s1_id'],row['cand_id'],hit))
    h=pl.DataFrame(rows,schema=['s1_id','cand_id','hypothesis'],orient='row')
    result={}
    for kind in h['hypothesis'].unique().sort():
        v=h.filter(pl.col('hypothesis')==kind)
        e=v.group_by('s1_id').len('removed_n').join(selected_counts,on='s1_id').with_columns(new_n=pl.col('old_n')-pl.col('removed_n'))
        # Exact upper bound for deletion-only changes: remaining predictions all true;
        # removed predictions false, and no unpredicted truths. Empty-after means singleton.
        e=e.with_columns(bound=pl.when(pl.col('new_n')==0).then(1.0).otherwise(1-1.25*pl.col('new_n')/(.25*pl.col('new_n')+pl.col('old_n'))))
        result[kind]={'pairs':v.height,'entities':e.height,'deletion_only_max_full_test_gain':float(e['bound'].sum()/1732544),
                      'new_singletons':e.filter(pl.col('new_n')==0).height}
        d.join(v,on=['s1_id','cand_id']).select('s1_id','cand_id','qname','tname','qaddr','taddr','hypothesis','same_name_number_s1s').write_parquet(O/(kind+'_examples.parquet'))
    # Bound for the entire extended noun bucket, including streets excluded above.
    v=d.filter(pl.col('cohort')=='noun_same_number')
    e=v.group_by('s1_id').len('removed_n').join(selected_counts,on='s1_id').with_columns(new_n=pl.col('old_n')-pl.col('removed_n'))
    e=e.with_columns(bound=pl.when(pl.col('new_n')==0).then(1.0).otherwise(1-1.25*pl.col('new_n')/(.25*pl.col('new_n')+pl.col('old_n'))))
    result['all_extended_nouns']={'pairs':v.height,'entities':e.height,'deletion_only_max_full_test_gain':float(e['bound'].sum()/1732544),'new_singletons':e.filter(pl.col('new_n')==0).height}
    (O/'residual_noun_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
