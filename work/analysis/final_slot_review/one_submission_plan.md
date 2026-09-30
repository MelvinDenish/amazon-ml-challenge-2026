# One available submission, one reserved: evidence review

27 September 2026. User confirmed v12 is **not submitted**. Two attempts remain and one must stay reserved. This review ran local diagnostics only; it did not train remote models, change production predictions, or upload anything.

## Decision

The best confirmed submission is **v11_up2_frnoun = 0.985752**. Exceeding 0.987 is a plausible stretch, but the completed experiments do not establish a reliable one-submission route. **0.99 is not supported by the current evidence.** V12 combines a small extension of a successful rule with a much larger unvalidated change in France model selection. It should not be described as a demonstrated 0.987–0.99 submission.

The highest-return remaining work is to resolve the **v11/v12 France disagreements using the scores already available**, separate the effects of the rule and stack, and build one coherent candidate. Additional broad model training, calibration sweeps, or candidate rebuilding have weaker measured returns.

## All recorded public scores

| Submission | Public score | Change from preceding main submission |
|---|---:|---:|
| v03_xgb_translit | 0.949713 | — |
| v04cand_xgb_extra | 0.962207 | +0.012494 |
| v05_stage2 | 0.959750 | -0.002457 |
| v06_pool600_frac50 | 0.966682 | +0.006932 |
| v07_blend_own | 0.967051 | +0.000369 |
| v07_ce1s_own | 0.978947 | +0.011896 |
| v07_ce12x_own | 0.982536 | +0.003589 |
| v09_ce5_frad_own | 0.983900 | +0.001364 |
| v10_up1_ce8_fr09plus | 0.983929 | +0.000029 |
| v11_up2_frnoun | **0.985752** | **+0.001823** |

The France-empty diagnostic scored 0.848752. It was a measurement, not an improvement candidate. Other files in the submission ledger have blank public scores and must not be treated as successful leaderboard experiments. In particular, v12 remains unsubmitted.

Several rows change multiple components: the v09 gain cannot be assigned wholly to extra encoders, and the v10 result cannot separately establish the effect of its recall push. The v10-to-v11 comparison is cleaner: this audit verified **zero changed India or US rows**, with 13,352 changed France rows.

## Size of the remaining target

Full test S1 counts are France 259,452, India 809,986, US 663,106; total 1,732,544. France's full-test share is 0.149752.

| Target | Additional public score needed | France-only improvement if public country weights equal full-test weights |
|---|---:|---:|
| 0.987 | +0.001248 | +0.008334 |
| 0.988 | +0.002248 | +0.015011 |
| 0.990 | +0.004248 | +0.028367 |

The final column is scenario arithmetic, not a measured France score or forecast. The public leaderboard scores an undisclosed subset. Moreover, the France-empty diagnostic equals the France singleton fraction on that subset, not zero. Without the public country weight and France singleton fraction, it does not uniquely identify country scores. The inconsistent France estimates of roughly 0.952–0.958 in old notes are estimates, not independently observed labels.

## What v12 actually changes

Both incumbent and pending submission SHA1 values match the ledger. Comparing final TSV sets, v12 changes **16,561 France entities**, adds **8,285 pairs**, and removes **9,073 pairs**. India and US are identical to the incumbent. There are 277 empty-to-nonempty rows and 429 nonempty-to-empty rows.

| Pair category | Added | Removed |
|---|---:|---:|
| Filler-word variants | 5,652 | 649 |
| Exact normalized core-name tokens | 1,350 | 2,524 |
| Extended same-number noun swaps | 0 | 2,110 |
| Noun swaps at another number / empty address | 5 | 141 |
| Other changes | 1,278 | 3,649 |

These are diagnostic groups defined in `audit_final_slot.py`, not ground-truth classes. Of the 1,350 exact-core-name additions, **910 have empty addresses**. Among the 3,649 other removals, sampled examples include acronyms and completely different names at the same address, as well as plausible sibling errors. The latter cannot all be assumed false. The positive public noun-rule result does not validate these separate model-stack decisions.

Only 352 of the 2,110 extended noun removals satisfy the original address-similarity threshold of 85. The remainder reflects the removal of the street condition as well as broader vocabulary. This is a new generalization of the rule, not a repeat of the already scored experiment.

For the 2,110 extended noun pairs alone, a deletion-only policy has a **maximum full-test macro gain of +0.000336252**, even granting perfect deletions. This uses each affected S1's actual predicted set size. If an old set has n predictions and a new set has m after deletion, its largest possible gain is 1 for m=0, otherwise `1 - 1.25*m/(0.25*m+n)`. The maximum assumes every retained prediction true, every removed prediction false, and no missed truths. The formula was checked by exhaustive truth-count enumeration for set sizes 1–12.

This bound covers only deleting those flagged pairs; it is not a bound on ownership changes, the whole v12 stack, or an unknown public subset. It is just 27% of the full-test-equivalent gain needed for 0.987.

## Fresh checks of tempting alternatives

* **Further noun-rule expansion is tiny.** A same-number/same-street noun conflict with one additional typo found 41 selected pairs; even perfect deletion is at most +0.0000058 on the full test. Close-spelling vocabulary swaps found 234, with a perfect-deletion bound of +0.0000354. Close spelling includes plurals and potentially legitimate variants, so blindly deleting them is not justified.
* **Narrow incremental retrieval was tested instead of assuming a rebuild helps.** Exact normalized core name, same first house number, unique reference for that key across all S1s, strong full-address similarity at least 95, candidate absent from the current pool, and no currently selected owner: 22 US validation pairs, all true, for +0.0000155 US F0.5; zero India pairs. France has only 50 such proposals across 43 entities. This narrow policy cannot carry the target. It does not prove all other retrieval strategies have the same ceiling.
* **The completed CE veto did not pass a strong low-harm gate.** At max-logit threshold -3, 15 held-out pairs were flagged and 7 were true. At -5 there were only four flags, all false. The -3 test run flagged 689 pairs total. Its implementation multiplies odds by 0.001, so a flag is not necessarily a deletion; full postprocessing is needed to evaluate it. This is not a credible source of a +0.001248 gain.
* **Supporting-record empty-address specialist:** measured macro changes -0.00001 India and -0.00002 US. No reason to repeat it unchanged.
* **Global calibration:** the selected adjustment lost on the other held-out half; identity was best on the second half.
* **More encoders / broad recall push:** the combined latest public improvement was +0.000029. Brand assignment and exact expected-F selection previously yielded only small local improvements. None establishes the needed step.

## Highest-ROI plan for the available attempt

1. **Freeze v11 as the anchor and keep India/US fixed.** The submitted France rule is the strongest recent causal evidence. The latest India/US validation scores are already 0.988136 and 0.987847, while their newest model additions produced small improvements. Preserve the reserved attempt; do not spend either attempt on another diagnostic empty-country upload.

2. **Separate the existing France changes before selecting the candidate.** Compare cached predictions for (a) v11 plus the noun extension only, (b) the multilingual France stack with the original noun rule, and (c) full v12. For each, recompute ownership and final sets and count additions, deletions, owner switches, and singleton transitions. Never sum gains from overlapping rules. Some of these predictions already exist (`test_v12_extA`, `test_v11_ml5fr_noun`, `test_v12_ml5fr_ext`), so this requires selection and analysis rather than another encoder run.

3. **Concentrate the audit on the largest unresolved categories.** The 5,652 filler additions are the clearest domain-specific hypothesis. Separate same-number/strong-street/uncontested records from weak-address or competing-owner cases; compare the exact same decision slice on labelled countries. The quoted 92.5% US analogue rate is for a broader group and is not measured precision for these newly added France records. Independently assess the 2,524 exact-core-name removals and 3,649 other removals: preserve useful acronym/alias matches unless the available owner and address evidence supports their rejection. Assess the 910 empty-address additions separately because the earlier empty-address specialist did not improve the metric.

4. **Choose a single France policy after those comparisons.** Compare full v12 against a selective combination that uses the multilingual stack only in supported categories and retains v11 elsewhere. Treat the selective combination as another hypothesis, not as an automatic winner. Use labelled, paired per-S1 macro changes and resampling by connected owner groups where feasible; audit France stability across distinct name families and S1 neighborhoods. Stability and synthetic tests do not supply missing France labels, so they cannot justify a claimed confidence interval for France leaderboard uplift.

5. **Spend the available attempt on the strongest integrated candidate, not merely to discover whether v12 is good.** Before upload, verify the final candidate subset and one-owner constraints, run the official validator, and verify unchanged India/US output. Keep the second attempt reserved. If the remaining gains are supported only by the +0.000336 maximum rule extension and unlabelled model guesses, the honest status remains “incremental candidate, 0.987 unproven”; do not describe it as a 0.99 plan.

There is no defensible numeric v12 forecast from the present evidence. The historical ledger's 0.9859–0.9864 estimate is an unvalidated prediction. Crossing 0.987 would require a material gain from the larger France stack disagreements in addition to the small noun extension; crossing 0.99 would require substantially more than anything established by the remaining experiments.

## Reproducible evidence

`audit_results.json` contains country counts, exact final-set differences, target arithmetic, and cohort totals. `residual_noun_results.json` contains deletion bounds. `incremental_recovery_results.json` contains the narrow retrieval pilot. Detailed added/removed pair tables and example tables are saved alongside this note. Scripts are `../audit_final_slot.py`, `../probe_france_residual_nouns.py`, and `../probe_incremental_recovery.py`. Production files and both submission TSVs were unchanged.
