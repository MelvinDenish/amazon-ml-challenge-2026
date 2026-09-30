"""Read-only experiment audit. Writes diagnostics under work/analysis only."""
from pathlib import Path
import gc
import json
import sys
import time

import polars as pl

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs
from score import per_entity_f05
from postprocess import one_owner, select_expected_f05

ART = WORK / 'artifacts'
OUT = WORK / 'analysis'
V6 = ART / 'k2_out_v2/artifacts'


def log(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def metric(gt, pred, ids):
    return per_entity_f05(gt, pred, ids)


def run():
    summary = {'data': {}, 'countries': {}, 'provenance': {
        'oof': str(V6 / 'oof/xgb_v6.parquet'),
        'root_candidate_cache': 'Older pool-150 candidates; not the v06 pool-600 artifact.',
        'v06_oracle': 'Computed from y=1 pairs in the exact saved v06 OOF table, on its sampled S1 population.',
        'limitations': 'OOF contains sampled S1s with candidates, globally learned transliteration, and sampled ownership competitors. France has no labels.'}}
    for split in ('train', 'test'):
        for source in ('source1', 'source2', 'source3'):
            counts = pl.scan_parquet(ART / f'data/{split}_{source}.parquet').group_by('country').len().collect()
            summary['data'][f'{split}_{source}'] = counts.to_dicts()
    log(summary['data'])
    gt_all = load_ground_truth_pairs()
    s1_all = pl.read_parquet(ART / 'data/train_source1.parquet', columns=['entity_id', 'country'])
    for country in ('India', 'US'):
        start = time.time()
        ids = s1_all.filter(pl.col('country') == country).rename({'entity_id':'s1_id'}).select('s1_id')
        gt = gt_all.join(ids, on='s1_id', how='semi')
        cands = pl.scan_parquet(ART / f'cands/train_{country}.parquet').select('s1_id', 'cand_id')
        found = gt.lazy().join(cands, on=['s1_id','cand_id'], how='semi').collect()
        ceiling = metric(gt, found, ids['s1_id'])
        cinfo = {'root_cache_blocking': {
            's1_total': ids.height,
            'true_pairs': gt.height,
            'candidate_pair_recall': found.height / gt.height,
            'candidate_macro_oracle': ceiling['f05'].mean(),
            'nonempty_s1_with_no_true_candidate': ceiling.filter((pl.col('n_true')>0)&(pl.col('tp')==0)).height,
            's1_with_missing_truth': ceiling.filter(pl.col('tp')<pl.col('n_true')).height,
            'singletons': ceiling.filter(pl.col('n_true')==0).height,
        }}
        log({'country': country, 'blocking':cinfo})
        missed = gt.join(found, on=['s1_id','cand_id'], how='anti')
        missed.write_parquet(OUT / f'{country}_root_cache_blocking_misses.parquet')
        del ceiling, cands
        oof = pl.scan_parquet(V6 / 'oof/xgb_v6.parquet').filter(pl.col('country')==country).collect()
        eval_ids = oof.select('s1_id').unique()
        gt_eval = gt.join(eval_ids, on='s1_id', how='semi')
        truth_candidates = oof.filter(pl.col('y')==1).select('s1_id','cand_id')
        missed = gt_eval.join(truth_candidates,on=['s1_id','cand_id'],how='anti')
        missed.write_parquet(OUT / f'{country}_v06_blocking_misses.parquet')
        oracle = metric(gt_eval, truth_candidates, eval_ids['s1_id'])
        owned = one_owner(oof, 'p')
        pred = select_expected_f05(owned, 'p')
        scores = metric(gt_eval, pred, eval_ids['s1_id']).with_columns(
            fp=(pl.col('n_pred').cast(pl.Int64)-pl.col('tp')),
            fn=(pl.col('n_true').cast(pl.Int64)-pl.col('tp')),
        ).join(oracle.select('s1_id',pl.col('f05').alias('oracle_f05'),pl.col('tp').alias('n_found')),on='s1_id')
        scores = scores.with_columns(
            pl.when(pl.col('f05')==1).then(pl.lit('perfect'))
            .when(pl.col('n_true')==0).then(pl.lit('singleton_false_merge'))
            .when((pl.col('fp')>0)&(pl.col('fn')>0)).then(pl.lit('both_fp_fn'))
            .when(pl.col('fp')>0).then(pl.lit('fp_only'))
            .otherwise(pl.lit('fn_only')).alias('error_type'))
        scores.write_parquet(OUT / f'{country}_v06_entity_scores.parquet')
        err = scores.group_by('error_type').agg(pl.len().alias('n_s1'),
            (1-pl.col('f05')).sum().alias('total_score_loss'),pl.col('fp').sum(),pl.col('fn').sum()).to_dicts()
        fp_pairs = pred.filter(pl.col('y')==0)
        fn_pairs = truth_candidates.join(pred.select('s1_id','cand_id'),on=['s1_id','cand_id'],how='anti')
        removed_correct = truth_candidates.join(owned.select('s1_id','cand_id'),on=['s1_id','cand_id'],how='anti').height
        perfect_precision = metric(gt_eval,pred.filter(pl.col('y')==1),eval_ids['s1_id'])['f05'].mean()
        cinfo.update({
            'oof_s1':eval_ids.height, 'oof_pairs':oof.height,
            'oof_macro':scores['f05'].mean(), 'oof_candidate_oracle':oracle['f05'].mean(),
            'v06_sample_candidate_pair_recall': truth_candidates.height / gt_eval.height,
            'v06_sample_blocking_missed_pairs': missed.height,
            'oof_if_remove_all_fp':perfect_precision,
            'oof_error_groups':err, 'fp_pairs':fp_pairs.height,
            'retrieved_but_missed_pairs':fn_pairs.height, 'true_pairs_lost_by_one_owner':removed_correct,
            'threshold_scores':{},
        })
        for threshold in (0.5,0.6,0.7,0.8,0.9):
            cinfo['threshold_scores'][str(threshold)] = metric(gt_eval,owned.filter(pl.col('p')>=threshold),eval_ids['s1_id'])['f05'].mean()
        # Calibrate/diagnose on stored OOF only, never test labels.
        bins = oof.with_columns((pl.col('p')*20).floor().clip(upper_bound=19).cast(pl.UInt8).alias('bin'))
        cinfo['calibration'] = bins.group_by('bin').agg(pl.len().alias('n'),pl.col('p').mean(),pl.col('y').mean()).sort('bin').to_dicts()
        sample = pl.concat([
            fp_pairs.sort('p',descending=True).head(50).select('s1_id','cand_id','p').with_columns(pl.lit('FP').alias('error')),
            oof.join(fn_pairs,on=['s1_id','cand_id'],how='semi').sort('p').head(50).select('s1_id','cand_id','p').with_columns(pl.lit('retrieved_FN').alias('error')),
            missed.sort('s1_id','cand_id').head(100).with_columns(pl.lit(None,dtype=pl.Float32).alias('p'),pl.lit('blocking_FN').alias('error')),
        ],how='vertical_relaxed')
        q = pl.scan_parquet(ART/'data/train_source1.parquet').rename({'entity_id':'s1_id','business_name':'q_name','business_address':'q_addr'}).select('s1_id','q_name','q_addr')
        target = pl.concat([pl.scan_parquet(ART/f'data/train_{s}.parquet') for s in ('source2','source3')]).rename({'entity_id':'cand_id','business_name':'t_name','business_address':'t_addr'}).select('cand_id','t_name','t_addr')
        sample.lazy().join(q,on='s1_id').join(target,on='cand_id').collect().write_csv(OUT/f'{country}_error_examples.tsv',separator='\t')
        # Join exact v06 feature shards by IDs, with separate retrieval/selection errors.
        base = pl.scan_parquet(V6/f'feats/train_{country}.parquet').select('s1_id','cand_id','t_a_empty','t_is_native','t_was_native','t_is_domain','n_tset','a_tset')
        extra = pl.scan_parquet(V6/f'feats_extra/train_{country}.parquet').select('s1_id','cand_id','form_disjoint','twin_flag','num_absdiff')
        labelled = oof.lazy().filter((pl.col('y')==1)|(pl.col('p')>0.05)).join(base,on=['s1_id','cand_id']).join(extra,on=['s1_id','cand_id'])
        labelled = labelled.join(pred.select('s1_id','cand_id').with_columns(pl.lit(1).alias('selected')).lazy(),on=['s1_id','cand_id'],how='left').with_columns(pl.col('selected').fill_null(0))
        group = pl.when(pl.col('form_disjoint')==1).then(pl.lit('form_conflict')).when((pl.col('twin_flag')==1)&(pl.col('num_absdiff')<=10)).then(pl.lit('near_number_twin')).when(pl.col('twin_flag')==1).then(pl.lit('other_number_twin')).when(pl.col('t_a_empty')==1).then(pl.lit('empty_address')).otherwise(pl.lit('rest'))
        cinfo['pair_error_groups'] = labelled.with_columns(group.alias('group')).group_by('group').agg(
            ((pl.col('selected')==1)&(pl.col('y')==0)).sum().alias('FP'),
            ((pl.col('selected')==0)&(pl.col('y')==1)).sum().alias('retrieved_FN'),
            ((pl.col('selected')==1)&(pl.col('y')==1)).sum().alias('TP'),
        ).collect().to_dicts()
        cinfo['elapsed_seconds'] = round(time.time()-start,1)
        summary['countries'][country] = cinfo
        (OUT/'audit_v06_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        log({'country':country,'results':cinfo})
        del oof,owned,pred,scores,oracle,bins,found,gt,gt_eval,fp_pairs,fn_pairs,truth_candidates
        gc.collect()


if __name__=='__main__':
    run()
