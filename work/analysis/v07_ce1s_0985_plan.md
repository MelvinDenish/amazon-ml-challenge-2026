# Evidence and execution plan after v07_ce1s_own = 0.978947

26 September 2026. Budget supplied by the user: two submissions remaining today and approximately 15–20 GPU hours. This review used the actual source, submission history, saved predictions, Kaggle logs, and fresh local diagnostics. Kaggle reported both `bedrock-k9c-xlmr` and `bedrock-k9d-ce3` RUNNING during the review. No new remote training job or leaderboard submission was started.

**Recommendation: finish the diverse encoder experiments already running, then focus new work on empty-address matching and targeted candidate recovery. Treat France transfer as a separate uncertainty. Another generic tree-model sweep, blanket veto, or probability-threshold sweep is a poor use of the remaining budget.**

The required improvement is **0.006053**, equal to **28.75% of the current public-score deficit**. No available evidence guarantees 0.985 or first place. The private labels and competitors' final scores are unknown. The plan below sets measurable promotion gates instead of promising a score.

**What is already implemented and what its evidence actually says**

| Component / experiment | Evidence | Decision |
|---|---|---|
| Rule baseline and initial XGBoost | Completed, superseded by stronger models | Preserve as provenance; no new budget |
| Transliteration, v03 | Label-learned native-to-Latin map; public 0.949713 | Keep; use fold-local learned maps in new strict validation |
| Legal forms, number relations, twin competition, v04 | Public 0.962207 | Keep useful signals; hard exclusions are not justified |
| Cluster stage 2, v05 | Local gain, public regression to 0.959750 | Do not revive unchanged |
| Pool 600, top 20 plus reverse top 3, more training, v06 | Public 0.966682 | Current retrieval baseline; recall and training-size effects are confounded |
| XGBoost variants plus ownership probabilities, v07 | Public 0.967051; blend weights 0.4 v6, 0.6 v7b | Keep as base scores |
| CatBoost | Worse local results; fitted blend assigns weight 0 | Do not train another generic CatBoost variant |
| Raw-number/unit/name-difference features, v08 | Implemented and evaluated; adding the same first CE gives 0.98570 versus 0.98566 for v07+CE in the earlier LR-stack comparison | Little demonstrated marginal value after CE; not the main next step |
| Distil multilingual CE v1, linear stack | Completed; local pooled 0.98566 | Confirms raw text adds information beyond similarities |
| CE v1 plus context stack, `ce1s` | Fresh reproduction: 0.986317 pooled; public **0.978947** | Preserve unchanged as incumbent |
| Larger Distil CE v2 plus v1, `ce12` | Fresh reproduction: **0.987391** pooled, +0.001074; existing output passes recorded validation; no public score supplied | Best completed successor in the saved labeled evaluation |
| XLM-R CE, `k9c` | Implemented; remote job RUNNING, no completed scores available locally at review time | Evaluate when finished; do not count an anticipated gain |
| Synthetic sibling-negative CE, `k9d` | Implemented; remote job RUNNING | Evaluate real-data gains and transfer before blending |
| Added-qualifier veto | Helped the older GBM; essentially redundant after `ce1s`, only about 0.00001–0.00003 in India and zero displayed US gain | No dedicated submission |
| EM label-shift correction | Historical India 0.978290→0.977690, US 0.981820→0.981420 | Reject unchanged |
| Cell-specific sibling prior correction | Test predictions exist; reported gains are expectations under adjusted probabilities, not labeled test results | Research candidate only; matching train counts is not validation |
| Pseudo-labeling under country transfer | US→India 0.9274→0.9277 | Weak return; not a primary budget allocation |
| Naive anchor/bridge recovery | Historical pilot deltas −0.000188 India, −0.000353 US; proposal precision 31–34% | Reject automatic transitive linking |
| Retrieval pilot | Forward top-40 recovers 0.49 percentage points of true pairs in India and 0.31 in US; name char-3 top-10 adds only 0.02/0.01 points | Test selective retrieval, not universal doubling or the same char route |

The retrieval pilot excludes production reverse-top-3 recovery and uses 30k queries per country. Its gains are not incremental production recall estimates. The local pilot initially ran out of memory, but the later Kaggle run completed; it would be wrong to call the entire experiment untested.

The older v06 plan is partly stale: deterministic feature sampling, sliced best-iteration models for newly trained boosters, raw compound-number features, and a repaired base pipeline driver now exist. Historical v6/v7 artifacts can still differ from current source. The current driver still does not reproduce the complete final CE stack, and the generic CE runner differs from the historical Kaggle scripts in its monitoring split and checkpoint handling.

**Fresh audit: where the present model actually loses points**

I replayed both saved CE stacks through exclusive-owner probabilities, one-owner selection, and the existing F0.5 set selector on the same **275,402 S1 entities**. This is historical validation, not a newly untouched test set. The following interventions use training truth and are *oracle upper bounds*, not implementable rules or predicted leaderboard scores.

| Historical evaluation | India: 110,187 S1 | US: 165,215 S1 | Pooled |
|---|---:|---:|---:|
| Current `ce1s` | 0.986247 | 0.986364 | **0.986317** |
| Completed `ce12` | 0.987481 | 0.987331 | **0.987391** |
| `ce1s`, remove every selected false positive | 0.988273 | 0.988319 | 0.988300 |
| `ce1s`, recover every retrieved empty-address true match | 0.990111 | 0.991563 | 0.990982 |
| Perfect classification of current candidates | 0.993799 | 0.995007 | 0.994524 |
| `ce1s`, fix all errors outside the original CE routing band | 0.986522 | 0.986699 | 0.986628 |

The pooled score weights this validation population, not the test country proportions. Oracle gains overlap and must not be added.

- `ce1s` misses **18,967 true pairs that are already candidates**. **14,547 (76.7%)** have an empty candidate address.
- `ce12` reduces retrieved misses to **17,521**, but **14,397 (82.2%)** still have empty addresses. It fixes only **150** of the earlier 14,547 empty-address misses. Scaling this same CE formulation mostly helped other cases.
- Another **15,894 true pairs are absent from the candidate set**. About **40% of the current historical deficit** remains even with perfect classification of current candidates.
- Current selected false positives total **1,919**. Perfect removal supplies only **+0.001983** pooled historical score. “F0.5 favors precision” is not a reason to ignore recall at this stage.
- Perfectly repairing all currently mistaken non-routed pairs supplies only **+0.000310**. Widening the uncertainty band across millions of easy pairs has limited measured return. Test distribution shift could change this; use a targeted risk sample before expanding it.

Pair counts are not the scoring objective: a missed only match costs far more macro score than one missing member of a large set. Rank error groups by per-entity score loss and use full-pipeline macro F0.5 for every promotion.

**The larger CE result is credible locally, but France deserves special treatment**

`ce12` improves 2,566 validation entities and worsens 670. The gains are +0.001234 in India and +0.000967 in US. Evaluation S1 IDs do not overlap CE v2 training S1 IDs. However, target records appear across owner competitions: 214,829 routed evaluation pairs reuse a target that occurs in CE v2 training. Some are training positives for other owners. This is record exposure, not proof that the reported gain is invalid.

On the diagnostic subset whose **routed evaluation targets are absent from CE v2 training**, `ce12` still gains +0.001200 in India and +0.000977 in US. This supports the improvement; it does not turn the historical stack into strict nested validation.

The saved test outputs change substantially by country:

| Country | Test S1 entities | Changed S1 sets, `ce1s`→`ce12` | Share changed | Removed / added pairs |
|---|---:|---:|---:|---:|
| India | 809,986 | 13,479 | 1.66% | 5,934 / 7,966 |
| US | 663,106 | 15,493 | 2.34% | 9,798 / 6,303 |
| France | 259,452 | 32,870 | **12.67%** | 23,013 / 12,664 |

France contributes about 15% of test entities, but more than half the changed entity sets. This does not prove the changes are bad. It proves that the public effect cannot be predicted by simply adding the US/India validation gain. Nor can the overall public score identify France's score. `min(India, US)` in the local test-weighted score is only a heuristic, not a France lower bound.

**A real implementation defect exists, but it is not a large free gain**

In `src/ce_stack.py`, both `ce_other_max` and `ce_cand_other_max` assign the group's second-largest CE value to every row. The strongest *other* value should be second-largest for the winning row and largest for non-winners. For scores `[4, 2, -1]`, the current feature gives `[2, 2, 2]`; the intended feature is `[2, 4, 4]`. The S1 version is wrong on 222,865 of the 411,798 routed evaluation rows.

I tested the correction in an isolated CPU refit with the same features, folds, and fitting procedure for both arms. Original pooled score: **0.986280**; corrected: **0.986332**, a gain of **0.000052**. India changes −0.000010 and US +0.000094. CPU refits do not exactly reproduce the saved GPU model, so compare these two controlled arms, not one arm against the stored score. Repair and evaluate the feature when refitting; do not spend a leaderboard attempt on this alone.

The selector also uses a ratio-of-expectations approximation to expected F0.5 and an independence approximation for the empty set. Exact expected-F experiments are reasonable CPU work after calibration, but unmeasured and low priority. General F-measure decision theory supports examining this distinction, not a claim of competition gain ([Waegeman et al.](https://arxiv.org/abs/1310.4849)).

**Execution order and promotion gates**

**1. Freeze the baseline and establish a reliable experiment gate.**

Preserve `v07_ce1s_own` and its SHA-1 `140cd7fc82839164e127e619993fcc2a16d3f120`. Persist the evaluation S1 universe and candidate keys before any new filtering. Include S1s with zero candidates. Freeze folds and a final-check subset before developing the new specialist. Existing evaluation has already informed many decisions, so explicitly label it as development data.

For new training, keep tightly defined competing-owner families together; assign target records to the same component as their labeled owner. Avoid grouping every common name into enormous components. Generate training contexts with out-of-fold predictions and keep validation labels out of learned transliteration, anchors, and hard-negative mining. Evaluate against all competitors in the fixed deployment-like universe. A fresh outer holdout requires all fitted components to respect it; merely declaring a new split after fitting does not create one.

Use a primary fixed development evaluation, a competing-family stress evaluation, and small genuine US→India / India→US transfer checks for new CE variants. A transfer model must be trained without the held-out country's labels; filtering a both-country-trained model's scores is not such a test. A small checkpoint-selection split should be separate from the final metric evaluation. Bootstrap paired score differences by the competition family, not individual correlated pairs.

Suggested budget gate: promote a nontrivial new component when full-pipeline development gain is at least **0.0003**, its paired uncertainty interval supports improvement, and there is no material regression on the other country or the family stress set. These are proposed engineering thresholds, not statistical guarantees. Use stricter evidence for changes that affect many French entities.

**2. Finish and compare the running CE models before launching another generic encoder.**

Use `ce12` as the completed challenger. Evaluate: `ce12`; `ce12 + XLM-R`; and `ce12 + XLM-R + synthetic CE`. Compare standalone errors and *incremental* ensemble gains. Do not average in a weaker model merely because it is different. Fit small, regularized stacks with the corrected context features and compare a simple calibrated logit blend as a control. Retain individual CE ranks and disagreements if useful rather than assuming their raw-logit scales are comparable.

Report empty-address, singleton, same-name/different-number, qualifier-change, and ambiguous-owner slices. Apply every model to the same candidate universe. In the synthetic experiment, a randomly shifted house number plus qualifier is only a constructed negative; true matches also contain number corruption. Control synthetic weight, retain real positive variations, and prefer the supplied ground-truth hard negatives. A model that simply learns “different number means negative” can repeat v05's transfer failure.

For France, use the country-transfer checks and inspect disagreement slices, not a desired matches-per-S1 count. Treat France-specific blending as a hypothesis without labeled France evidence. If the larger CE remains unstable there, retaining incumbent France predictions while changing supported countries is an explicit conservative fallback, not proof of better France accuracy.

**3. Train one focused empty-address / ambiguous-owner specialist. This is the highest-priority new model.**

Start from a licensed multilingual encoder already in use. Build batches from the real retrieved empty-address positives, their highest-scoring wrong owners, distractors, and matched controls. Preserve labels from the supplied ground truth. Include alias, abbreviation, native-script, and generic-name examples; do not manufacture negatives just because a name is shared.

Represent the candidate's raw name and the S1 record, with **at most one or two independently high-confidence anchor names** from that S1's existing predicted matches. Provide explicit field labels and missing-address markers. The anchor's useful contribution is often a known alternate name. An absent address supplies no evidence to invent. Choose anchor confidence using held-out precision, and create training anchors out of fold using the same inference procedure. Never feed a truth-picked anchor into training while using noisy predicted anchors at test time.

Compare three bounded variants: name/pair only; pair plus predicted anchor names; and the winner with owner competition. Use all plausible competing S1s plus a **null-owner option**. A candidate's correct owner is not always among retrieved S1s, so null calibration matters. Retain singleton safeguards and calibrate on naturally distributed groups after any oversampling. Train a small gate for this slice and preserve the general CE elsewhere.

Stop if the specialist cannot improve end-to-end macro F0.5 with its actual predicted anchors. The prior bridge experiment establishes that unscored transitive propagation is unsafe here. Some empty-name/address combinations remain unidentifiable from the inputs; the oracle +0.004664 is headroom, not achievable promised gain.

**4. Pilot targeted retrieval against the exact production candidate set.**

Use two fixed query samples with the full country's S2/S3 target pool. Start with entities with no strong match, generic/aliased names, or missing fields. Compare incremental routes: adaptive forward top-40 for ambiguous queries; address-first retrieval retaining compound numbers/locality; and retrieval through names of predicted high-confidence anchors. Consider a small multilingual name retriever only if these pilots show that semantic alias coverage, rather than reranking, is the bottleneck.

Intersect every proposal with the current candidate keys to measure *new* recovered truth. Measure macro candidate-oracle gain, newly recoverable entities, candidate overhead, and runtime. Do not force ground-truth matches into validation candidate sets. A practical continuation gate is **at least +0.0008 macro oracle gain** with modest overhead (initial cap about five additional candidates per S1 averaged over the full population), followed by **at least +0.0003 actual macro gain** after scoring. Otherwise stop and keep the existing blocker.

Every new edge needs matching-model scores from a model trained/calibrated for that retrieval route. Add route/rank indicators and representative new hard negatives; reconstruct competition features. New candidate edges are outside the existing GBM-uncertainty routing mechanism, so route them explicitly through the CE. Regenerate `candidate_pairs.tsv` from the full edge set actually scored. Reusing v06's file after adding candidates would be incorrect.

**5. Combine only the successful components; calibrate and package once.**

Run a compact ablation matrix: incumbent; best general CE ensemble; plus specialist; plus retrieval. Select by exact per-S1 macro F0.5 after full assignment. Inspect singleton false merges and cases where one wrong owner steals another S1's only true match. A set-utility-aware ownership refinement is an optional later probe; arbitrary reassignment is not part of the default plan.

Keep normalization, candidate files, sampled IDs, folds, model/checkpoint hashes, feature order, calibration, and submission hashes in a manifest. Save CE and stacker models; the generic scripts do not currently save every fitted component. Package the actual training/preparation/selection code, including any synthetic or specialist step, rather than only the current generic README. Run the official validator, assert every S1 appears once, verify target ownership consistency, and verify that the candidate file equals the scored edge universe.

**GPU budget and the two submissions**

These are spending caps, not promised runtimes; confirm remaining quota including already-running jobs. Use CPU for cached-prediction audits, stack fitting, and most retrieval pilots.

| Work | Initial GPU allowance | Stop / extend rule |
|---|---:|---|
| Finish existing XLM-R and synthetic CE jobs, including their inference | up to 5 h remaining | Count their actual remaining charges first; no duplicate launches |
| One empty-address specialist pilot and winning fit | 4 h | Stop if predicted-anchor validation fails |
| Small country-transfer screens / focused rescoring | 2 h | Prefer small pilots; no full model grid |
| Final winner inference and integration | 2 h | Reserve before spending on optional work |
| Contingency | 2 h | Preserve for failures, rerun, packaging verification |
| Optional extension if 20 h is confirmed | up to 5 h | Extend the winning specialist or useful retrieval, not an unvalidated model family |

If existing jobs consume more than their cap, reduce optional experiments first. The CPU retrieval pilot is time-boxed separately; only its accepted extra-edge scoring may consume the focused-rescoring allowance. The earlier Distil CE v2 run completed training and 3.55M test-pair inference in about 1.78 hours of recorded job time, but XLM-R, longer inputs, hardware, and quota accounting can differ.

**Submission 1:** the strongest completed general CE challenger after the above gates. `v07_ce12_own` is the current ready fallback; prefer the finished XLM-R blend only if it actually improves the comparisons. Use one coherent, validated change and record its exact hash. Do not spend this slot on the tiny context fix or redundant veto.

**Submission 2:** the strongest integrated general CE + specialist + any accepted retrieval change. Reserve enough wall time for inference and official validation. If the new components fail their gates, retain the best scored artifact; the second submission is optional. One public score is a check on an aggregate hypothesis, not permission to make repeated France-specific threshold adjustments. Keep the public/private distinction in the final model choice.

The development objective is to move toward **0.990+ with improved transfer evidence**. Even that is not an LB conversion rule: the present historical-to-public gap is about 0.00737. If that gap remained unchanged, merely reaching local 0.990 would still miss public 0.985. Reducing the transfer gap matters alongside local gains. A route to the requested score needs multiple complementary gains; existing `ce12` alone has not established it.

The field-preserving text and difficult-example approach is consistent with established entity-matching practice ([Ditto](https://arxiv.org/abs/2004.00584)); its published benchmark improvements do not predict this competition. No external business lookup or external identity data is needed or recommended.

**Reproducible evidence produced in this review**

- `audit_v07_ce.py` and `audit_v07_ce_results.json`: model replay, candidate ceiling, oracle interventions, routing, and error groups.
- `probe_ce_cohorts.py` and `ce_cohort_results.json`: target exposure cohorts and exact test-submission differences.
- `probe_ce_context.py` and `ce_context_probe_results.json`: isolated CPU comparison of original versus corrected other-max features.
- Per-country/per-model entity-score parquet files support paired analysis.

Run the scripts from the repository root with the installed Python environment; the audit writes only under `work/analysis`. Historical findings are also traceable to `submission_log.csv`, `residual_errors.log`, `qualifier_rule_ce.log`, `labelshift_v06.log`, the bridge probe, and Kaggle k4/k5/k6/k8/k9 logs. Original source, trained artifacts, and submissions were preserved during this review.
