# Business Entity Resolution: Amazon ML Challenge 2026 (Team Bedrock)

This pipeline links every Source-1 (S1) business to its matching Source-2/Source-3 records.

1. **Normalisation.** Names and addresses are cleaned rule-based (junk wrappers, legal forms, abbreviations, accents, number formats). Native-script tokens go through a native-script→Latin token map. That map is learned **only from the train ground truth**.
2. **Blocking and candidate cascade** (the candidate set is small per S1).
   - An inverted index over 8 key types (house number + street, name tokens, name prefix, sorted name, rare address tokens, address bigrams, name-token pairs, number + name token), partitioned by country. Keys with too many postings are dropped, so the index scales linearly with the number of records; S1s are processed in chunks.
   - The top 600 targets per S1 by key score are re-ranked with rapidfuzz, keeping the top 20 per S1 plus the top 3 S1s per S2/S3 record (about 26 per S1).
   - **Cascade filter:** the XGBoost pair model (step 4) discards blocked pairs with p ≤ 0.005. This is also the cross-encoder routing threshold, so these pairs could never be selected.
   - Result: **4.96 candidates per S1** (median 5, 99th percentile 11), down from 26.6 per S1 after blocking. This is the set scored by the cross-encoders, the stacker and the final assignment, and it is written to `candidate_pairs.tsv`.
   - On 275k held-out S1s, the filter keeps 99.96% of the true pairs that blocking found, and F0.5 is unchanged (India 0.98814, US 0.98785 → 0.98784).
3. **Pair features.** About 75 features in total:
   - String similarity: rapidfuzz ratios for names and addresses.
   - Competition and context: rank, margins, how many S1s compete for a record.
   - Legal forms, house-number relations, twin competition.
   - Raw-address compound numbers and units, and name tokens present on only one side.
4. **XGBoost pair classifier.** CUDA, 3-fold GroupKFold by S1. It is a blend of two variants (depth 9, and depth 11 with colsample 0.6), with weights chosen on the out-of-fold (OOF) predictions.
5. **Cross-encoders on uncertain pairs** (0.005 < p < 0.995).
   - Transformers read both **raw** records ("name ; address") and are fine-tuned on the train labels. Eight models are used, all MIT or Apache-2.0 and at most 560M parameters:

     | Tag | Model | Licence, size | Held-out logloss |
     |---|---|---|---|
     | v1 | `distilbert-base-multilingual-cased`, 337k pairs | Apache-2.0, 134M | 0.205 |
     | v2 | `distilbert-base-multilingual-cased`, 1.47M pairs | Apache-2.0, 134M | 0.167 |
     | xlmr, xlmr2 | `xlm-roberta-base`, 1 and 2 epochs | MIT, 278M | 0.162 / 0.157 |
     | mdeb, mdebf | `microsoft/mdeberta-v3-base`, 1M pairs and all pairs | MIT, 278M | 0.155 / 0.152 |
     | xlmrl | `xlm-roberta-large` | MIT, 560M | 0.155 |
     | debl | `microsoft/deberta-v3-large` | MIT, 435M | 0.150 |

   - This is the biggest single gain. The GBM only sees similarity numbers. A cross-encoder reads the actual words, so it separates look-alike "sibling" businesses ("<name> Holdings / Riverside / Développement" at a nearby number), which are frequent in test.
6. **France adaptation** (France has no labels; `ce_france.py`).
   - Four multilingual cross-encoders (v2, xlmr, xlmr2, xlmrl) continue training on a mix of three sources:
     - 300k train pairs;
     - France test pairs that the current stack labels with confidence (p ≥ 0.99 as positives, p ≤ 0.01 among the top-5 as negatives);
     - 55k synthetic French sibling negatives.
   - Only their **France** test scores replace the base scores (`ce_test_scores_<tag>fr.parquet`). US/India inference is unchanged.
   - **Round 2** (`ce_france_r2.py`), after the France rules were confirmed on the leaderboard:
     - The pseudo-labels come from the final predictions of the validated recipe. Co-located noun swaps are negatives, and the generator's filler variants are positives.
     - The synthetic siblings use only pure sibling qualifiers and legal forms at a nearby number.
     - The set is 600k original pairs plus 388k France pairs.
     - Trained models: XLM-R-large on Modal A100/H100 (`modal_ce.py`); XLM-R-base (continued from epoch 2) on Kaggle; mDeBERTa-v3-base on Modal H100.
     - Their France scores enter the France stacks (step 11).
     - Leaderboard, with US/India rows unchanged: 0.986229 → **0.986586** from one round-2 model.
7. **Stacker** (`ce_stack.py`). A small XGBoost combines the GBM probability, the cross-encoder logits and the cross-encoder context inside the S1 (rank, best other candidate, best other owner).
   - It is cross-fitted on 275k held-out S1s that no model was trained on. Held-out F0.5: India 0.9881, US 0.9879.
   - House-number, twin and legal-form features are deliberately **not** used here, because their train-calibrated effect shifts on test.
   - **US/India** use the 8-model stack.
   - **France** starts from the 3-model stack (v1, v2, France-adapted XLM-R; `--no-src-feats`) with a +1 logit push.
   - For the generator's filler variants ("X SARL Développement" for "X Santé SARL": same number and street, and no S1 carrying the record's exact name), France takes the higher p of that stack and of the multilingual France-adapted 5-model stack (v1, v2, xlmr, xlmr2, xlmr-large). These variants are true copies: the US analogue of this slice is 99.5% true.
   - The English `deberta-v3-large` is kept out of France because it rejects these French variants (`france_final.py`).
8. **France co-located sibling rule** (`swap_rules.py`, `predict.py --noun-cap France`).
   - France has many separate businesses that share a name prefix and an address but carry a different activity noun ("Gospel Sportive SARL" / "Gospel Club SARL"). For 43–66% of such records, another S1 carries exactly the record's name at that number. For the filler words the figure is 0–15%.
   - US and India have almost none of these (≈2 per 1k S1, 1% true), so the models never learned them and give them p ≈ 0.26 in France.
   - The rule caps p at 0.003 for same-number pairs whose names differ by one S1-vocabulary noun. Filler words, stopwords, typos and tokens shorter than 4 letters are excluded.
   - It moves US/India held-out F0.5 by −0.00005 / −0.00002.
   - Public leaderboard, with US/India rows byte-identical: 0.983929 → 0.985752 for the rule, then 0.986229 for the extended rule plus the filler step.
9. **Assignment and selection.**
   - Exclusive-owner probabilities: a record belongs to at most one S1 or to none, so the odds are normalised with a null owner.
   - Global one-owner assignment.
   - Per-S1 set selection that maximises expected F0.5.

**Rules compliance:**
- No external data, lookups or APIs.
- Pretrained models are only the encoders listed above: MIT/Apache-2.0 licensed and ≤ 8B parameters.
- The unlabelled test data is used only by the pipeline itself: candidate generation, features, inference, and the France self-training (the model's own confident predictions as pseudo-labels, no human labels).

## Environment
- Python 3.12–3.14. Run `pip install -r requirements.txt`.
- CPU steps (normalisation, blocking, features) ran on a 4-core laptop and on Kaggle CPU sessions (4 vCPU, 30 GB RAM).
- GPU steps:
  - XGBoost on a Kaggle P100.
  - Base-size cross-encoders on Kaggle T4s. Training takes about 1.3h for DistilBERT and about 3h for XLM-R-base on 1.5M pairs; scoring the 3.5M test pairs takes about 40 min.
  - The large models (`xlm-roberta-large`, `deberta-v3-large`, full-data `mdeberta-v3-base`) and the France-adapted XLM-R-large ran through `modal_ce.py` on one A100-80GB (Modal), about 1.5h each. Data moved through presigned URLs.

## Data location
By default the code expects:
```
Amazon_ML/
  student_resource/student_resource/dataset/{train,test}/*.tsv
  work/code/business_entity_resolution/   <- this folder
```
Override with `BER_DATA_DIR` (folder holding `train/` and `test/`), `BER_ARTIFACT_DIR` (intermediate files), `BER_OUTPUT_DIR`, and `BER_DEVICE` (`cuda` or `cpu`).

## Reproduce
From `src/`, in order:

| Step | Command | Output |
|---|---|---|
| 1 | `python -c "from io_utils import convert_all_to_parquet; convert_all_to_parquet()"` | `artifacts/data/*.parquet` |
| 2 | `python translit.py` | `artifacts/translit_map.parquet` |
| 3 | `python build_candidates.py --split all` | `artifacts/cands/` (pool 600 → top 20 + reverse top 3) |
| 4 | `python features.py --split train --frac 0.5`, then `--split test` | `artifacts/feats/` (deterministic 50% S1 sample) |
| 5 | `python extra_features.py --split train`, then `--split test` | `artifacts/feats_extra/` |
| 6 | `python train_gpu.py --name xgb_v6 --extra` and `python train_gpu.py --name xgb_v7b --extra --depth 11 --colsample 0.6 --lr 0.06 --seed 7` | models, OOF |
| 7 | `python ensemble.py blend --names xgb_v6,xgb_v7b --out blend_v7`, then `python predict.py --name blend_v7 --blend --version base` | `oof/blend_v7.parquet`, `oof/test_blend_v7.parquet` |
| 8 | `python ce_pairs.py --eval-oof blend_v7 --train-oofs xgb_v6,xgb_v7b --test test_blend_v7` | `artifacts/ce/ce_{train2,eval,test}.parquet` |
| 9 | `python cross_encoder.py --model <hf model> --train ce_train2.parquet --tag <tag> [--lr ..]` for the base models (GPU); `modal run --detach modal_ce.py --tag <tag> --model <hf model> --urls <urls.json>` for the large ones | `artifacts/ce/ce_{eval,test}_scores_<tag>.parquet` |
| 10 | `python ce_france.py`, then continue training v2 / xlmr / xlmr2 / xlmrl on `ce_train_fr.parquet` (`modal_ce.py ... --fr 1` for xlmrl). Replace the France rows of each test score file with the adapted model's scores. Round 2: `python france_final.py ... --out test_final` and `--prune 0 --out test_final_nocascade`, then `python ce_france_r2.py`, train on `train_r2.parquet` (`modal_ce.py --tag xlmrlfr2 --model xlm-roberta-large`, `--tag mdebr2 --model microsoft/mdeberta-v3-base`, and `--tag xlmrlfr3 --fr 1` with the full train set plus the France part) and score `test_fr.parquet` | `ce_test_scores_<tag>fr.parquet`, `ce_{eval,test}_scores_<tag>r2.parquet` |
| 11 | US/India: `python ce_stack.py --base-oof blend_v7 --base-test test_blend_v7 --ce v1=ce_eval_scores.parquet:ce_test_scores.parquet --ce v2=... --ce xlmr=... --ce xlmr2=... --ce mdeb=... --ce mdebf=... --ce xlmrl=... --ce debl=... --out test_v10_ce8`. France base stack: `--ce v1=ce_eval_scores.parquet:ce_test_scores.parquet --ce v2=ce_eval_scores_v2.parquet:ce_test_scores_v2.parquet --ce xlmr=ce_eval_scores_xlmr.parquet:ce_test_scores_xlmrfr.parquet --ce xlmr2=ce_eval_scores_xlmr2fr2.parquet:ce_test_scores_xlmr2fr2.parquet --ce mdeb=ce_eval_scores_mdebr2.parquet:ce_test_scores_mdebr2.parquet --ce xlmrl=ce_eval_scores_xlmrlfr3.parquet:ce_test_scores_xlmrlfr3.parquet --no-src-feats --out test_fr_base`. France filler stack: the same list with `v2=ce_eval_scores_v2.parquet:ce_test_scores_v2fr.parquet` and without `--no-src-feats`, `--out test_fr_ml` | held-out F0.5 per country, `oof/test_{v10_ce8,fr_base,fr_ml}.parquet` |
| 12 | `python france_final.py --main test_v10_ce8 --fr-base test_fr_base --fr-ml test_fr_ml --out test_final` (cascade filter `--gbm test_blend_v7 --prune 0.005`, France +1 logit push, filler step, extended co-located sibling cap) | `oof/test_final.parquet` (8.6M pairs) |
| 13 | `python predict.py --name final --reuse --owner-prob --version final` (reads `oof/test_final.parquet`) | both submission TSVs, validated with `--check-ids` |

The final version is `v16_r2_all` (public LB **0.98674**). It uses the France stacks of step 11 with the round-2 models.
- The previous version, `v15_r2_xlmrl` (public LB **0.986586**), used a smaller France base stack: v1, v2, xlmr-fr, and XLM-R-large round 2 trained on 600k pairs (`ce_*_scores_xlmrlfr2`).
- `v13_int_frnounext_filler` (0.986229) used v1, v2 and xlmr-fr only. Running with `--prune 0` reproduces its predictions exactly.
- The cascade (`--prune 0.005`, default) shrinks the candidate file from 26.6 to 4.96 pairs per S1.

`run_pipeline.py` chains steps 1–7. The Kaggle job scripts under `work/kaggle_jobs/` show the exact commands that produced each submission.

## Validation
- **OOF:** 3-fold GroupKFold by S1 for the GBM, scored with the **full pipeline** (one-owner + expected-F0.5 set selection) using the exact macro F0.5 (`score.py`).
- **Cross-encoder and stacker:** measured on 275k S1s that are held out from both the GBM samples' folds and the cross-encoder training.
- **Test-side, label-free check:** `diagnose.py` counts, per country, the selected pairs in "suspicious" groups (legal-form conflict, near-number twin, empty address) and the matches per S1, and compares them with train. These counts called the leaderboard direction correctly for every version.

## Source files
| File | Purpose |
|---|---|
| `config.py` | Paths, constants, device |
| `io_utils.py` | TSV/parquet I/O, submission writer |
| `normalize.py`, `translit.py` | Normalisation; learned native-script→Latin token map |
| `dataset.py` | Loads one country's S1 (queries) and S2+S3 (targets) |
| `blocking.py`, `build_candidates.py` | Multi-key blocking, rapidfuzz re-rank, recall report |
| `features.py`, `extra_features.py` | Pair features; legal-form, house-number, twin and raw-address features |
| `train_gpu.py`, `train.py` | XGBoost (CUDA) with GroupKFold; OOF evaluation |
| `ensemble.py` | Second model family and OOF-weighted blending |
| `ce_pairs.py`, `cross_encoder.py`, `ce_stack.py` | Cross-encoder pair lists, fine-tuning and scoring, stacking |
| `ce_france.py`, `modal_ce.py` | France adaptation set (confident pseudo-labels + synthetic French siblings); remote A100 runner for the large encoders |
| `swap_rules.py`, `france_final.py` | France co-located sibling rule and filler variants; assembly of the final France predictions |
| `postprocess.py` | Exclusive-owner probabilities, one-owner assignment, expected-F0.5 selection |
| `predict.py`, `submission.py` | Test inference, writing, validation and logging |
| `diagnose.py`, `labelshift.py` | Error groups and test-side counts; label-shift check (measured, not used) |
| `score.py` | Exact macro-F0.5 metric (self-test: README example = 0.714) |
| `run_pipeline.py` | End-to-end driver for the GBM part |
