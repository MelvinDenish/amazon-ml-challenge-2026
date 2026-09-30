"""Audit saved CE predictions without modifying models or submissions.

Run from the repository root: python -u work/analysis/audit_v07_ce.py
All oracle interventions are diagnostic upper bounds, never deployable rules.
"""
from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import polars as pl

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'code/business_entity_resolution/src'))
from io_utils import load_ground_truth_pairs
from postprocess import exclusive_owner_prob, one_owner, select_expected_f05
from score import per_entity_f05

ART, OUT = WORK / 'artifacts', WORK / 'analysis'
KEY = ['s1_id', 'cand_id']
MODELS = {
    'ce1s': 'test_blend_v7_ce1s_evaloof.parquet',
    'ce12': 'test_blend_v7_ce_v1v2_evaloof.parquet',
}


def emit(x):
    print(json.dumps(x, ensure_ascii=True), flush=True)


def run():
    result = {'limitations': [
        'Historical S1-held-out CE/stacker evaluation; not a new nested or country-held-out test.',
        'Base OOF preprocessing and competitor universe have historical leakage/coverage limitations.',
        'Oracles use train truth only. Test and France labels are unavailable.',
        'Probability routing is measured using original blend_v7 p, not final stacked p.',
    ], 'countries': {}, 'paired_comparison': {}}
    gt = load_ground_truth_pairs()
    for country in ('India', 'US'):
        start = time.time()
        base = (pl.scan_parquet(ART / 'oof/blend_v7.parquet').filter(pl.col('country') == country)
                .filter((pl.col('s1_id').hash(seed=45) % 10000) < 5000)
                .select(*KEY, pl.col('p').alias('base_p')).collect())
        ids = base['s1_id'].unique()
        truth = gt.join(pl.DataFrame({'s1_id': ids}), on='s1_id', how='semi')
        group = (pl.scan_parquet(ART / f'groups/train_{country}.parquet')
                 .select(*KEY, 't_a_empty', 'twin_flag', 'form_disjoint', 'num_absdiff')
                 .join(base.lazy().select(KEY), on=KEY, how='semi').collect())
        country_result = {'n_eval_s1': len(ids), 'models': {}}
        allscores = {}
        for tag, filename in MODELS.items():
            o = (pl.scan_parquet(ART / 'oof' / filename)
                 .filter(pl.col('country') == country).collect())
            assert o.select(KEY).unique().height == o.height
            assert set(o['s1_id']) == set(ids)
            pred = select_expected_f05(one_owner(exclusive_owner_prob(o, 'p'), 'p'), 'p').select(KEY)
            found = o.filter(pl.col('y') == 1).select(KEY)
            fp = pred.join(truth, on=KEY, how='anti')
            fn = found.join(pred, on=KEY, how='anti')
            missed = truth.join(found, on=KEY, how='anti')
            scores = per_entity_f05(truth, pred, ids)
            allscores[tag] = scores.select('s1_id', pl.col('f05').alias(tag))
            scores.write_parquet(OUT / f'{country}_{tag}_entity_scores.parquet')
            score = float(scores['f05'].mean())
            def metric(p):
                return float(per_entity_f05(truth, p, ids)['f05'].mean())
            nofp = pred.join(truth, on=KEY, how='semi')
            diag = {'macro_f05': score, 'pred_pairs': pred.height, 'false_positive_pairs': fp.height,
                    'retrieved_false_negative_pairs': fn.height, 'blocking_missed_pairs': missed.height,
                    'true_pair_recall': found.height / truth.height,
                    'candidate_oracle': metric(found), 'remove_all_FP_oracle': metric(nofp),
                    'add_all_retrieved_FN_oracle': metric(pl.concat([pred, fn])),
                    'add_all_blocking_FN_oracle': metric(pl.concat([pred, missed]))}
            entity = scores.with_columns(
                (pl.col('n_pred') - pl.col('tp')).alias('fp'),
                (pl.col('n_true') - pl.col('tp')).alias('fn')).with_columns(
                pl.when(pl.col('f05') == 1).then(pl.lit('perfect'))
                .when(pl.col('n_true') == 0).then(pl.lit('singleton_FP'))
                .when((pl.col('fp') > 0) & (pl.col('fn') > 0)).then(pl.lit('FP_and_FN'))
                .when(pl.col('fp') > 0).then(pl.lit('FP_only')).otherwise(pl.lit('FN_only')).alias('error'))
            diag['entity_errors'] = entity.group_by('error').agg(pl.len().alias('n_s1'),
                ((1 - pl.col('f05')).sum() / len(ids)).alias('macro_loss')).sort('macro_loss', descending=True).to_dicts()
            errors = pl.concat([fp.with_columns(pl.lit('FP').alias('error')),
                                fn.with_columns(pl.lit('FN').alias('error'))])
            e = errors.join(base, on=KEY).join(group, on=KEY, how='left').with_columns(
                ((pl.col('base_p') > .005) & (pl.col('base_p') < .995)).alias('routed'),
                pl.when(pl.col('t_a_empty') == 1).then(pl.lit('empty_address'))
                .when(pl.col('form_disjoint') == 1).then(pl.lit('form_conflict'))
                .when((pl.col('twin_flag') == 1) & (pl.col('num_absdiff') <= 10)).then(pl.lit('near_twin'))
                .when(pl.col('twin_flag') == 1).then(pl.lit('other_twin')).otherwise(pl.lit('rest')).alias('group'))
            diag['error_routing'] = e.group_by('error', 'routed').len().sort('error','routed').to_dicts()
            diag['error_groups'] = e.group_by('group', 'error').len().sort('group','error').to_dicts()
            outside = e.filter(~pl.col('routed'))
            repaired = pl.concat([pred.join(outside.filter(pl.col('error') == 'FP').select(KEY), on=KEY, how='anti'),
                                  outside.filter(pl.col('error') == 'FN').select(KEY)])
            diag['perfect_nonrouted_errors_oracle'] = metric(repaired)
            empty_fn = e.filter((pl.col('error') == 'FN') & (pl.col('group') == 'empty_address')).select(KEY)
            diag['recover_empty_address_FN_oracle'] = metric(pl.concat([pred, empty_fn]))
            if tag == 'ce1s':
                evpairs = (pl.read_parquet(ART / 'ce/ce_eval_scores.parquet')
                           .join(base.select(KEY), on=KEY, how='semi'))
                ce_tr = pl.read_parquet(WORK / 'kaggle_jobs/bedrock_ce/ce_train2.parquet', columns=KEY)
                overlap = evpairs.select('cand_id').unique().join(ce_tr.select('cand_id').unique(), on='cand_id')
                exposed = evpairs.join(overlap, on='cand_id', how='semi')
                diag['CE_v2_eval_target_overlap'] = {'unique_target_ids': overlap.height, 'eval_pairs': exposed.height,
                    'eval_S1s': exposed['s1_id'].n_unique(), 'note': 'Target reuse across different S1s, not necessarily positive-label leakage.'}
                x = evpairs.with_columns(
                    pl.col('ce').sort(descending=True).slice(1, 1).first().over('s1_id').alias('old'),
                    pl.col('ce').max().over('s1_id').alias('mx'))
                x = x.with_columns(pl.when(pl.col('ce') == pl.col('mx')).then(pl.col('old')).otherwise(pl.col('mx')).alias('correct'))
                diag['wrong_S1_other_max_rows'] = x.filter(pl.col('old') != pl.col('correct')).height
                e.write_parquet(OUT / f'{country}_ce1s_pair_errors.parquet')
            country_result['models'][tag] = diag
            emit({'country': country, 'model': tag, **diag})
            del o, pred, found, fp, fn, scores, e
            gc.collect()
        pair = allscores['ce1s'].join(allscores['ce12'], on='s1_id').with_columns((pl.col('ce12') - pl.col('ce1s')).alias('delta'))
        d = pair['delta'].to_numpy()
        result['paired_comparison'][country] = {'mean_delta': float(d.mean()), 'improved_S1': int((d > 1e-10).sum()),
            'worsened_S1': int((d < -1e-10).sum()), 'paired_SE_S1_assuming_independence': float(d.std(ddof=1) / np.sqrt(len(d))),
            'positive_delta_sum': float(d[d > 0].sum()), 'negative_delta_sum': float(d[d < 0].sum())}
        result['countries'][country] = country_result
        (OUT / 'audit_v07_ce_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        emit({'country': country, 'seconds': round(time.time()-start, 1), 'comparison': result['paired_comparison'][country]})
        del base, group, truth, allscores
        gc.collect()
    emit({'saved': str(OUT / 'audit_v07_ce_results.json')})


if __name__ == '__main__':
    run()
