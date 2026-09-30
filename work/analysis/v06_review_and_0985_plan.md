# v06 review and a measured route toward 0.985

Date: 26 September 2026. Baseline public leaderboard: **0.966682**.

## Recommendation

Keep v06 as the baseline. The strongest next experiment is a **richer pair matcher with better address parsing and explicit competing-owner information**, followed by a **small multilingual text reranker trained on actual hard negatives**. Expand retrieval selectively and use learned cluster evidence only if it passes an independent validation test.

Do not expect a different XGBoost depth, another probability threshold, naive pseudo-labeling, or an unconditional graph merge to supply the required gain. Reaching 0.985 requires +0.018318, or removing **54.98% of the current deficit from a perfect score**. Nothing in the available artifacts establishes that this gain is achievable on the hidden test labels. The plan below is a prioritized way to pursue it, with measurable reasons to continue or stop.

## What was reviewed and rerun

- The supplied challenge README and guidelines PDF, current source modules, v06 code snapshot, submission history, Kaggle launch scripts, and available training/error logs.
- The exact v06 OOF prediction artifact, covering **27,261,839 pairs and 1,103,306 S1 entities**. I recomputed the metric, candidate oracle, threshold sweep, calibration bins, and error counts using the full ground truth for those S1s.
- A native XGBoost inference probe on 227,354 pairs from 8,994 India S1s.
- Address parsing probes derived from actual false-positive examples.
- A conservative recovery experiment using predicted anchors for empty-address targets; it failed and is excluded from the recommended submission changes.

This was analysis, not a new training/submission run. Production source, existing submission files, and leaderboard entries were not changed. The new files are confined to `work/analysis/`.

### Artifact provenance matters

The main `work/artifacts/cands/` directory is still the older pool-150 candidate set. For example, its India candidate table contains 21,569,210 pairs and has 96.37% pair recall. The pool-600 India artifact has 22,296,505 pairs; the Kaggle log reports 98.07% recall. The main cache and its pool-150 backup have matching file sizes and row counts.

For v06 measurements, this review uses:

- `work/artifacts/k2_out_v2/artifacts/oof/xgb_v6.parquet`
- Corresponding `feats/`, `feats_extra/`, and `models/` under that same artifact root.
- The v06 test probabilities are separately under `work/artifacts/k3_out/artifacts/oof/test_xgb_v6.parquet`.

The JSON audit explicitly separates the older root-cache blocking results from the v06 candidate oracle. A complete pool-600 US candidate file was not found locally; the v06 OOF pairs provide the actual candidate set for the evaluated S1 sample.

## Experiment history: what worked

| Experiment | Stored test-weighted OOF proxy | Public LB | Interpretation |
|---|---:|---:|---|
| v01 rules | about 0.748 | Not recorded | Weak baseline |
| v02 XGBoost | about 0.962 | Not recorded | Learned matching helped substantially |
| v03 transliteration | about 0.969 | 0.949713 | Script-aware retrieval helped India |
| v04 legal forms / numbers / twins | 0.9749 | 0.962207 | Distinguishing nearly identical businesses transferred well |
| v05 cluster stage 2 | 0.9767 | 0.959750 | Local gain did not transfer; LB fell 0.002457 |
| v06 pool 600, 50% train | 0.9796 | 0.966682 | Broader retrieval and more training data helped; LB rose 0.004475 over v04 |

v06 changed both candidate coverage and training size, so its improvement cannot be assigned to either factor alone. The earlier one-third-training learning-curve run scored about 0.9730 against v04's 0.9749 proxy: additional data helped, but this does not establish another two leaderboard points of scaling headroom.

The completed held-out-country pseudo-label experiment went from **0.9274 to 0.9277** on US-to-India transfer. Positive pseudo-label precision was 0.9851. This is weak evidence for spending more of the remaining quota on the same method.

The workspace contains a v07 XGBoost/CatBoost blend job and label-shift code, but no completed v07 scored result was found. These are planned experiments, not proven improvements.

## The measured v06 error budget

All numbers here describe the existing OOF population, with its validation limitations described below. They are not measurements of France or hidden test performance.

| Quantity | India | US |
|---|---:|---:|
| S1 entities evaluated | 441,582 | 661,724 |
| Macro F0.5 | **0.978292** | **0.981820** |
| True-pair candidate recall | 0.980744 | 0.985330 |
| Perfect matching within existing candidates: macro F0.5 | **0.994010** | **0.995061** |
| True pairs absent from candidates | 29,453 | 33,594 |
| Retrieved true pairs rejected | 44,834 | 60,911 |
| False-positive pairs selected | 9,295 | 10,242 |
| Retrieved true links lost at `one_owner` | 12,328 | 22,145 |
| Singleton S1s falsely given matches | 1,244 | 1,383 |

Across the evaluated population:

- Current macro F0.5: **0.980408**.
- Perfect classification of existing candidates: **0.994640**.
- Approximately **27.4% of the validation score deficit** is already unavoidable with the current candidates. The remaining **72.6%** is matching/assignment/selection error.
- Removing every selected false positive, while recovering no missed true match, would produce **0.985910**. This is an oracle intervention, not an attainable threshold. Even perfect false-positive removal leaves substantial missed-match loss.
- Correcting only the falsely matched singletons could add at most **0.002381** on this OOF population.

**Pair recall is not the macro-F0.5 ceiling.** A 98% candidate pair recall can support more than 98% macro F0.5 because missing one member of a larger cluster incurs a smaller loss. The older methodology document's blocking upper-bound statement is stale for v06.

### Where retrieved errors occur

These are mutually exclusive pair buckets; the ownership counts above overlap them and must not be added again.

| Error bucket | India FP | India retrieved FN | US FP | US retrieved FN |
|---|---:|---:|---:|---:|
| Empty target address | 932 | 19,981 | 1,764 | 38,082 |
| Nearby-number twin | 168 | 746 | 352 | 1,738 |
| Other-number twin | 567 | 1,773 | 592 | 2,513 |
| Disjoint legal forms | 9 | 29 | 139 | 1,159 |
| Other | 7,619 | 22,305 | 7,395 | 17,419 |

**58,063 / 105,745 = 54.9% of retrieved false negatives have empty addresses.** A further **32.6% of all retrieved false negatives** are removed during owner selection. These findings justify an owner-aware matcher and an empty-address specialist, but do not justify forcing additional matches.

There are genuine ambiguity cases. Among target records whose raw address is empty or a common null marker, 2.2% in India and 5.3% in the US share an exactly normalized full name with records belonging to another S1 or with distractors. This is only one ambiguity diagnostic, not a bound on the optimal score; approximate-name ambiguity is additional.

### Cheap ideas tested during this review

1. **Threshold sweep:** 0.7 gives 0.978569 India / 0.981969 US, only about +0.0002 overall over the current expected-F selector. Raising the threshold to 0.9 makes both countries worse. A global threshold is not the missing solution.
2. **Exact-name cluster support:** require at least two accepted nonempty-address records, each with OOF p >= 0.999, spanning S2 and S3; require a unique anchor owner for the normalized full name; add only currently unmatched empty-address targets on existing candidate edges. The added links were **31.15% correct in India and 34.39% in the US**. Macro F0.5 fell **0.000188 / 0.000353**. This rejects simple name-based transitive expansion, even with confident anchors. The hard residuals contain convincing distractors.
3. **Best-iteration inference:** replaying the correct fold at its best iteration reproduced stored OOF probabilities to a maximum absolute difference of 0.0000023. Including the extra rounds changed 1,211 probabilities by more than 0.01, but did not hurt the small diagnostic's aggregate score. The mismatch should be corrected for consistency; it is not evidence of a large free score gain.

## Concrete problems to fix before another expensive run

### 1. Address normalization discards distinguishing information

`normalize.py` retains only one house-number token and one following token as the street feature. It does not represent a full compound number or unit independently.

| Raw address fragment | Current `a_num` | Current `a_street` |
|---|---|---|
| Flat No A-5/101 | 5 | 101 |
| Flat No A-5/114 | 5 | 114 |
| 30-25-4/1 | 30 | 25 |
| 30-25-4/4 | 30 | 25 |
| 4421 1A Amethyst Court | 4421 | 1 |
| 4421 12A Amethyst Court | 4421 | 12 |

These patterns were found in false-positive examples. The current fuzzy address similarities retain some information, but the dedicated numeric features fail to identify the relevant disagreement.

Add raw-preserving compound-number, building, unit, floor, numeric-sequence, street-token, locality, and postcode features. Keep extraction confidence and alternatives when parsing is ambiguous. Five-digit postcode patterns matter for France/US; `features.py` currently extracts only six digits. Preserve leading zeros in postcodes. Do not treat a postcode or a floor ordinal as the house number. Do not hard-reject every number disagreement: legitimate truncations and corruptions exist.

For names, retain full names, extracted aliases, legal forms, initials, rare-token differences, and character-edit features. A token-set ratio of 1.0 is weak evidence when one name contains an extra discriminating token. A legal-form addition and a legal-form conflict are different signals; the data contain true matches in the conflict bucket, so blanket rejection is also wrong.

### 2. Validation does not measure France generalization

France is **14.975% of test S1s**. `score.py` substitutes `min(US_score, India_score)` for its score. That is a heuristic; calling it pessimistic does not make it a lower bound.

As an illustration only: if US and India test scores equaled their OOF scores, v06's overall LB would imply a France score of about **0.892**. This is NOT an estimate of actual France performance; it demonstrates why the overall leaderboard cannot identify which country is responsible for the gap. Known-country test shift may also be substantial.

Validation weaknesses:

- The transliteration map is learned from all training labels before splitting. Held-out labels therefore influence preprocessing and retrieval. Use fold-local maps for an honest validation estimate. Learning from all labels is appropriate for the final test model.
- Folds are hashes of S1 IDs. Similar competing businesses and shared target records can cross folds. Add a holdout grouped by tightly defined competing-business families or high-confidence graph components, and inspect component sizes to avoid giant generic-name groups.
- The owner competition used in OOF scoring contains only the sampled S1 population. Feature context was calculated on the full candidate table, but the final owner-selection operation is not performed over all S1s. Its effect is not uniformly optimistic or pessimistic; reproduce deployment competition in a fixed holdout universe.
- S1s absent from the candidate table are absent from evaluation. Define the evaluation S1 universe before blocking and include zero-candidate entities.
- Stage 2 uses stage-1 OOF probabilities, which is better than in-sample scores. However, ordinary OOF stacking is not a strict nested test: some stage-2 training features depend on base models trained on the outer validation labels. Use an untouched outer holdout or nested construction when evaluating a new stack.

Use three checks: a fixed entity holdout, a difficult-family holdout with all competing S1s, and both US-to-India and India-to-US transfer tests. In a strict country-transfer test, the target country's labels must also be excluded from learned normalization. None proves France performance, but together they can reject fragile improvements.

### 3. Reproduction and inference differ from the documented v06 run

- `train_gpu.py` predicts OOF using `iteration_range=(0, best_iteration + 1)`. The v06 snapshot and current `predict.py` use unrestricted native `Booster.predict` at test time.
- The saved fold models contain 1,675 / 1,868 / 1,918 rounds, while the selected best models contain 1,575 / 1,768 / 1,818 rounds. There are 100 extra rounds in every model. Use the same iteration range at train validation and test inference. This behavior is documented by [XGBoost](https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping).
- `features.py` samples from an unsorted `unique()` result. A random seed alone does not guarantee identical sampled IDs when the input order changes. Persist the S1 sample and folds, or use a deterministic hash rule with pinned library versions. Rebuilding 50% samples independently is dangerous for OOF blending.
- `ensemble.py` inner-joins OOF tables. If models cover different S1 samples, the evaluation silently shrinks to their intersection. Require identical IDs/candidates or explicitly evaluate the agreed fixed universe.
- `run_pipeline.py` defaults to 30% training and does not run `extra_features.py` or supply `--extra`; that command does not reproduce the documented v06 50%/66-feature model.
- `make_zip.py` packages current source rather than resolving the source that generated an existing prediction artifact. Preserve the existing v06 snapshot and package future runs with their manifests.

Give every run a manifest with candidate paths/hashes, normalization/map version, feature names/order, sampled IDs, folds, model parameters, best iterations, calibration version, and output checksums. Assert that predictions and the submitted candidate file cover exactly the scored edges.

## Proposed next model

### A. Improve retrieval through independent routes

Keep pool-600 blocking as a baseline. Add character n-gram retrieval separately for names and addresses, with raw and normalized views. Union those candidates with compound-number/street keys and a true target-to-S1 retrieval pass. The current reverse top-3 is applied AFTER the S1 pool cut, so it cannot rescue an edge that never entered that pool.

Use a pilot on a fixed 20k-50k S1 subset, searching the full target corpus. Compare top-20, top-40, and adaptive expansion for low-confidence queries. Do not multiply the candidate count for every easy query without measuring value.

Aim for >99.5% positive-pair recall and >0.997 macro candidate oracle on the development holdout as engineering goals, not predictions. Measure each route's unique recovered true links, additional negatives, runtime, and macro oracle gain. Expand into a larger full run only if the unique gain is material. Retrain matching on the union so newly retrieved negatives are represented.

### B. Train against competing owners, including no owner

For each S2/S3 record, assemble its plausible S1 owners plus a null-owner option. Train an owner-aware head or reranker using:

- v06 probability/logit and the new string/address features;
- the best other owner, margin, and which discriminating tokens or numbers favor each owner;
- source and missing-field patterns;
- robust support and contradiction from other records, using out-of-fold anchors;
- exact and fuzzy name frequency, including competing S1s outside the training subsample.

Supervise the correct owner or null using provided labels. Include all retrieved true owners for training analysis; if the owner is missing from retrieval, report that separately and do not silently call it a distractor. If injected training positives are used, label that intervention and keep validation retrieval natural.

Mining should focus on *actual labeled negatives*: same name/different unit, same address/different legal identity, one-letter name changes, dropped-address distractors, and the model's high-confidence mistakes. Preserve enough ordinary pairs for stable calibration. Synthetic augmentations may change punctuation/order/format while preserving a known label; do not assume that an invented one-digit or legal-form change is always a real negative.

Apply at most one owner per target, but keep the null option and allow many targets per S1. Compare assignments using marginal expected per-S1 F0.5 gain, rather than dropping all but the highest pair probability before set selection. On small ambiguous components, test a constrained local search or exact solver; there is no need to solve one giant graph. A one-to-one Hungarian assignment would be inappropriate because S1 may have multiple records from each source.

### C. Add a text reranker where it can contribute new information

Start with a small multilingual cross-encoder such as `google-bert/bert-base-multilingual-cased`, whose model card lists Apache-2.0 and multilingual coverage. Fine-tune only on supplied labels. This is a candidate to test, not a claimed superior model. [Model card](https://huggingface.co/google-bert/bert-base-multilingual-cased)

Serialize both records with field labels, retaining raw and normalized names, raw addresses, parsed numeric parts, and legal forms. Start around 192-256 tokens. Train a bounded pilot of approximately 100k-300k pairs, stratified over positives, hard negatives, empty addresses, scripts, and countries. Use actual nearby competing businesses as negatives. Benchmark the first 5k examples before committing the quota.

The rationale is that text retains distinctions lost by scalar fuzzy scores. Fine-tuning a pretrained encoder for pair classification is established in entity matching; [Ditto](https://arxiv.org/abs/2004.00584) is a relevant method, but its published results do not predict this competition's score.

At inference, route uncertain pairs AND structurally risky high-confidence pairs: compound-number conflicts, changed initials or legal identity, transliteration failures, and crowded owner competition. The observed high-confidence false positives mean that routing only p between 0.1 and 0.9 would miss some important errors. Benchmark and cap the reranked volume; do not blindly run a transformer over every candidate pair.

Blend the cross-encoder signal with engineered features and owner context on an independent calibration split. Keep it only if it adds full-pipeline macro F0.5 after calibration and assignment. Semantically similar businesses are often distinct, so semantic similarity alone is unsafe for this metric.

### D. Treat cluster support and singleton prediction as learned signals

The bridge pilot rejected unconditional propagation. If a learned member-consistency model is attempted, train S2/S3 pair evidence from the supplied entity clusters, exclude the candidate itself from its support features, include conflicting owners, and calibrate on untouched hard families. Never merge an entire connected component solely because one strong edge links it.

Add a per-S1 no-match/cardinality model using complete candidate evidence. Compare its no-match probability against the current product of independent pair nonmatch probabilities. The current expected-F calculation is a ratio-of-expectations approximation; exact small-set expected-F or an independently validated set model can be tested later. These are refinements after fixing the inputs and owner discrimination, not the main 0.0183-point bet.

## How to spend the remaining 30 Kaggle GPU-hours

These are quota caps, not promised runtimes or a claim that 30 clock hours remain. Much of candidate building is CPU work. The supplied guidelines put the challenge deadline at 27 September 2026, 11:59 PM IST; schedule submissions separately from GPU quota.

| Work package | GPU-hour cap | Continue only if |
|---|---:|---|
| Reproducible baseline, address features, honest holdout, small ablations | 4 | Replay is stable and features help the difficult-family holdout |
| Retrieval pilot + owner-aware XGBoost | 6 | Candidate oracle improves, and new negatives do not erase full-pipeline gains |
| Multilingual cross-encoder pilot and selective reranking | 10 | It improves macro F0.5 on independent difficult/transfer validation within measured throughput budget |
| Train the selected model configuration and perform final inference | 6 | Earlier ablations identify a clear winner |
| Reserve for failure recovery, calibration, output validation | 4 | Preserve this reserve until the winning pipeline is established |

The local GPU is an RTX 3050 Laptop with 6 GB VRAM. Only about 1 GB was free at inspection, so do not assume it can host a training job immediately. Use local CPU for analysis and bounded feature/retrieval probes; use Kaggle for the main training. Colab is not necessary to start this plan.

An illustrative prioritization threshold is at least +0.001 macro F0.5 on a fixed development set before scaling an expensive idea. It is not a universal significance cutoff: report paired, family-level bootstrap uncertainty, subgroup regressions, and the number of changed S1 predictions. Use an untouched final holdout after choosing configurations. Record India, US, singleton, empty-address, twin, and script-specific changes rather than selecting solely by one average.

For a credible 0.985 attempt, I would want a development trajectory toward about 0.99 macro F0.5 with expanded candidates and materially reduced transfer/competition gaps. This is a development gate, not a formula translating CV to LB. If the gains are small, keep v06 or the best validated successor instead of submitting a speculative aggressive version.

## What to deprioritize

- More depths/seeds on essentially the same information as the main experiment. A small final blend can still be useful.
- Repeating the same pseudo-label method after a measured +0.0003 transfer result.
- Blanket increases to the probability threshold.
- Unconditional same-name cluster recovery; it failed this review's pilot.
- EM prior correction as a cure for the entire domain shift. It assumes conditional feature distributions stay stable; new countries and changed corruption patterns need not satisfy that assumption.
- Huge dense embeddings or a generative LLM over all pairs before a small text pilot demonstrates value.
- Tuning decisions against a handful of public-LB movements. The private evaluation remains unseen.

The supplied rules prohibit external business-identity lookup and external data augmentation and require a final model within the stated license/size limits. The proposed record features, negatives, and labels come from the provided dataset; the external references here concern algorithms and model documentation only.

## Reproduce and inspect this review

From the workspace root:

```powershell
python -X utf8 work/analysis/audit_v06.py
python -X utf8 work/analysis/probe_v06.py
python -X utf8 work/analysis/probe_bridges.py
```

Outputs:

- `audit_v06_results.json`: recomputed metrics, calibration bins, error groups, and provenance.
- `probe_v06_results.json`: inference consistency, parser examples, and exact-name ambiguity diagnostic.
- `bridge_probe_results.json`: failed recovery-pilot measurements.
- `India_error_examples.tsv`, `US_error_examples.tsv`: selected actual FP, retrieved FN, and v06 blocking FN examples. These are diagnostic examples, not random population samples.
- `*_v06_entity_scores.parquet`: per-S1 score, false positives/negatives, and candidate oracle.
- `*_v06_blocking_misses.parquet`: missing truth links for the exact evaluated v06 S1 population.

The existing OOF uses globally learned preprocessing and sampled competition; the bridge experiment has no independent tuning split. These scripts diagnose that historical pipeline and must not be presented as newly leakage-free validation. **No new leaderboard improvement, including 0.985, has been demonstrated by this review.**
