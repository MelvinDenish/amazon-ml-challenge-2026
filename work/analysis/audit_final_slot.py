"""Read-only audit of scored incumbents and pending France submissions."""
from pathlib import Path
import sys
import json
import gc
import hashlib
import polars as pl
from rapidfuzz import fuzz, process

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'code/business_entity_resolution/src'))
from io_utils import load_records
from swap_rules import TOK, NUM, ADDR, FILLERS, STOP

OUT = W / 'analysis/final_slot_review'
OUT.mkdir(exist_ok=True)
KEY = ['s1_id', 'cand_id']

def read(version):
    return (pl.read_csv(W / 'submissions' / version / 'matching_results.tsv', separator='\t',
                        quote_char=None, missing_utf8_is_empty_string=True)
            .rename({'source1_entity_id': 's1_id', 'matched_entity_ids': 'ids'})
            .with_columns(pl.col('ids').fill_null('')))

def pairs(d):
    return d.filter(pl.col('ids') != '').select('s1_id', pl.col('ids').str.split(',').alias('cand_id')).explode('cand_id')

def main():
    q = load_records('test', 'source1')
    meta = q.select(pl.col('entity_id').alias('s1_id'), 'country')
    nc = meta.group_by('country').len().sort('country').to_dicts()
    result = {'country_counts': nc, 'n_s1': q.height, 'public_subset_country_weights_unknown': True,
              'current_best_score': .985752, 'v12_is_unsubmitted': True, 'comparisons': {}}
    base = read('v11_up2_frnoun').join(meta, on='s1_id')
    frbase = base.filter(pl.col('country') == 'France')
    pb = pairs(frbase)
    pb.write_parquet(OUT/'france_baseline_pairs.parquet')
    for version in ('v10_up1_ce8_fr09plus', 'v11_up3_ml5fr_noun', 'v12_up3_ml5fr_nounext'):
        d = read(version).join(meta, on='s1_id').rename({'ids': 'new'})
        z = base.join(d.select('s1_id', 'new'), on='s1_id')
        summary = z.group_by('country').agg(n=pl.len(), changed=(pl.col('ids') != pl.col('new')).sum()).sort('country').to_dicts()
        fr = d.filter(pl.col('country') == 'France').rename({'new': 'ids'})
        pn = pairs(fr)
        add = pn.join(pb, on=KEY, how='anti')
        rem = pb.join(pn, on=KEY, how='anti')
        add.write_parquet(OUT / (version + '_added.parquet'))
        rem.write_parquet(OUT / (version + '_removed.parquet'))
        changed = z.filter((pl.col('country') == 'France') & (pl.col('ids') != pl.col('new')))
        hist = changed.with_columns(
            old_n=pl.when(pl.col('ids') == '').then(0).otherwise(pl.col('ids').str.count_matches(',') + 1),
            new_n=pl.when(pl.col('new') == '').then(0).otherwise(pl.col('new').str.count_matches(',') + 1),
        ).group_by('old_n', 'new_n').len().sort('len', descending=True)
        hist.write_csv(OUT / (version + '_set_sizes.csv'))
        result['comparisons'][version] = {'country_row_changes': summary, 'France_added': add.height,
            'France_removed': rem.height, 'France_changed_S1': changed.height,
            'empty_to_nonempty': changed.filter((pl.col('ids') == '') & (pl.col('new') != '')).height,
            'nonempty_to_empty': changed.filter((pl.col('ids') != '') & (pl.col('new') == '')).height,
            'set_size_changes': hist.to_dicts()}
        print(json.dumps({'version': version, **{k:v for k,v in result['comparisons'][version].items() if k != 'set_size_changes'}}), flush=True)
        del d,z,fr,pn,changed
        gc.collect()

    q = q.filter(pl.col('country') == 'France').select(pl.col('entity_id').alias('s1_id'),
        pl.col('business_name').alias('qname'), pl.col('business_address').alias('qaddr'),
        TOK.alias('qt'), NUM.alias('qn'), ADDR.alias('qa'))
    r = pl.concat([load_records('test', s, 'France') for s in ('source2','source3')]).select(
        pl.col('entity_id').alias('cand_id'), pl.col('business_name').alias('tname'),
        pl.col('business_address').alias('taddr'), TOK.alias('tt'), NUM.alias('tn'), ADDR.alias('ta'))
    # Include pending additions and removals to explain what changed.
    v12add = pl.read_parquet(OUT/'v12_up3_ml5fr_nounext_added.parquet')
    v12rem = pl.read_parquet(OUT/'v12_up3_ml5fr_nounext_removed.parquet')
    universe = pl.concat([pb,v12add]).unique().join(q,on='s1_id').join(r,on='cand_id')
    counts = q.select(pl.col('qt').explode().alias('w')).group_by('w').len()
    vocab = counts.filter(pl.col('len') >= 30)['w'].implode()
    twins = q.select(pl.col('qt').list.join(' ').alias('name_key'),pl.col('qn').alias('tn')).group_by('name_key','tn').len('same_name_number_s1s')
    universe = universe.with_columns(add=pl.col('tt').list.set_difference('qt'),miss=pl.col('qt').list.set_difference('tt'),
        name_key=pl.col('tt').list.join(' ')).join(twins,on=['name_key','tn'],how='left').with_columns(
        pl.col('same_name_number_s1s').fill_null(0),
        a=pl.col('add').list.first(),m=pl.col('miss').list.first())
    ns = process.cpdist(universe['a'].fill_null('').to_list(),universe['m'].fill_null('').to_list(),scorer=fuzz.ratio,workers=-1)
    ads = process.cpdist(universe['qa'].to_list(),universe['ta'].to_list(),scorer=fuzz.token_set_ratio,workers=-1)
    universe = universe.with_columns(word_sim=pl.Series(ns),addr_sim=pl.Series(ads))
    noun = ((pl.col('add').list.len()==1)&(pl.col('miss').list.len()==1)&pl.col('a').is_in(vocab)&pl.col('m').is_in(vocab)
            &~pl.col('a').is_in(FILLERS['France']+STOP)&~pl.col('m').is_in(FILLERS['France']+STOP)&(pl.col('word_sim')<70)
            &(pl.col('a').str.len_chars()>=4)&(pl.col('m').str.len_chars()>=4))
    universe = universe.with_columns(
        same_num=(pl.col('qn').is_not_null()&(pl.col('qn')==pl.col('tn'))).fill_null(False),
        empty_t=(pl.col('taddr').fill_null('').str.strip_chars().str.len_chars()==0),
        noun_swap=noun.fill_null(False),
        filler_swap=((pl.col('add').list.len()==1)&(pl.col('miss').list.len()<=1)&pl.col('a').is_in(FILLERS['France'])).fill_null(False),
        exact_tokens=(pl.col('qt')==pl.col('tt')))
    universe = universe.with_columns(cohort=pl.when(pl.col('noun_swap')&pl.col('same_num')).then(pl.lit('noun_same_number'))
        .when(pl.col('noun_swap')&pl.col('empty_t')).then(pl.lit('noun_empty_address'))
        .when(pl.col('noun_swap')).then(pl.lit('noun_other_number'))
        .when(pl.col('filler_swap')).then(pl.lit('filler'))
        .when(pl.col('exact_tokens')).then(pl.lit('exact_core_tokens'))
        .otherwise(pl.lit('other')))
    for name,p in [('baseline_selected',pb),('v12_added',v12add),('v12_removed',v12rem)]:
        u = universe.join(p,on=KEY,how='semi')
        stat=u.group_by('cohort').agg(pl.len(),pl.col('s1_id').n_unique().alias('n_s1'),
            (pl.col('same_name_number_s1s')>0).mean().alias('has_S1_name_number_twin'),
            pl.col('addr_sim').mean().alias('mean_addr_sim')).sort('cohort').to_dicts()
        result[name]=stat
        print(json.dumps({name:stat}),flush=True)
        u.select(*KEY,'qname','tname','qaddr','taddr','cohort','same_num','addr_sim','same_name_number_s1s').write_parquet(OUT/(name+'_details.parquet'))
        u.filter(pl.col('noun_swap')).select(*KEY,'qname','tname','qaddr','taddr','cohort','same_name_number_s1s').head(150).write_csv(OUT/(name+'_noun_examples.tsv'),separator='\t')
    nfr=q.height
    result['target_gaps']={str(target):{'global_gap':target-.985752,'France_gain_if_only_France_changes_using_full_weight':(target-.985752)/(nfr/result['n_s1'])} for target in (.987,.988,.99)}
    (OUT/'audit_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'country_counts':nc,'target_gaps':result['target_gaps']}),flush=True)

if __name__=='__main__':main()
