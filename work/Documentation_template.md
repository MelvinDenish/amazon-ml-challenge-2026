# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Bedrock
**Team Members:** Melvin Denish, JeiKarthikPandi, Sneha Manikandan, Ahamed Vifaaq
**Submission Date:** 27-09-2026

---

## 1. Executive Summary
We treat the task as four steps: blocking, pairwise scoring, stacking, and set selection.

1. **Blocking with a learned cascade.** A multi-key inverted index proposes candidates, a fuzzy name/address re-rank keeps about 26 per Source-1 (S1) entity, and a learned cascade filter cuts that to **4.96 candidates per entity** while keeping 98.0–98.5% of the true pairs.
2. **GPU XGBoost classifier.** It scores every pair with about 75 similarity and competition features.
3. **Cross-encoders on uncertain pairs.** Eight fine-tuned multilingual and English transformers (all MIT/Apache, ≤ 560M parameters) re-score the uncertain pairs from the raw text. A cross-fitted XGBoost stacker combines them.
4. **Metric-aligned selection.** Exclusive-owner probabilities, a global one-owner assignment, and per-entity expected-F0.5 set selection.

France has no labels. For it we add two things:
- self-training of the multilingual encoders on the model's own confident French predictions;
- a rule for France-specific co-located sibling businesses, derived from structure measured in the data.

**Scores:**
- Held-out macro F0.5 on 275k unseen S1s: India 0.9881, US 0.9879.
- Public leaderboard: **0.98674** for the final version (v16). The score was 0.9839 before the France work.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Structure of the training labels.**
  - 2.21M S1s, 3.46 matches on average (range 0–11), 5.6% singletons.
  - Every S2/S3 record has **at most one** S1 owner.
  - Matches never cross countries.
  - 26% of S2/S3 records are distractors that match no S1.
- **Noise.**
  - Typos, legal-suffix swaps, word reordering and junk wrappers ("(ID: 96415)", "***").
  - Domains and handles, and alias forms ("f/k/a", "t/a").
  - About 11% of names are in Indic script.
  - Addresses: reordered components, "##801"-style numbers, abbreviations, `<NULL>` placeholders, about 3% empty.
  - A "filler" transformation drops the last noun and appends a generic word after the legal form ("Womens Health Partners LLC" → "Womens Health LLC Center"). It is 92.5% true in US.
- **Hard negatives.**
  - 40% of S1 names are shared by another S1.
  - Sibling businesses are the S1 name plus a qualifier ("Holdings", "Group", "Riverside") at a nearby house number, often as their own 2–3 record cluster. They are about 0% true.
- **Train→test shift.** We measured it without labels by comparing train and test statistics.
  - Test has 23% more records per S1, and sibling patterns are 3–9× more frequent.
  - France (15% of S1) is unseen. Its S1 names are generic ("<prefix> <noun> SARL"), 9.2% of France S1s share name and number with another S1 (US 0.9%), and co-located businesses are common.

### 2.2 Solution Strategy
**Approach type:** blocking + GBM + cross-encoder stack + graph-constrained expected-F0.5 selection.

**Core innovations:**
1. **Cross-encoders on raw text, applied only to uncertain pairs.** They read which word differs between the records, which similarity scores cannot express. Leaderboard: 0.967 → 0.979 with one encoder, 0.9825 with three, 0.9839 with five.
2. **Exclusive-owner probabilities.** o = p/(1−p), and q_s = o_s/(1 + Σo), so that competing S1s and the "no owner" option share one record.
3. **Unlabelled-country adaptation.**
   - Self-training on confident French pseudo-labels, together with synthetic French sibling negatives.
   - A data-derived France sibling rule.
   - A second round of self-training, whose labels come from the leaderboard-validated rules. It adds +0.00036 on the leaderboard with a single model.
4. **One diagnostic upload that separated France from US+India.** We uploaded v09 with every France row emptied (LB 0.848752). That places France at ≈ 0.958 and US+India at ≈ 0.9885, which matches our held-out score. After that, every France change was judged with that decomposition.

---

## 3. Candidate Generation (Blocking)
- **Normalisation.** Accent stripping, junk and alias removal, canonical legal suffixes and street types, and number normalisation. Indic tokens are mapped to Latin with a token map learned **only from train labels**.
- **Stage 1: inverted index within the country.** Eight key families:
  - house number + street
  - name token
  - 10-character no-space name prefix
  - sorted name tokens
  - rare address token
  - address bigram
  - name-token pair
  - number + name token

  Frequent keys are capped. The score is Σ weight / log2(1 + document frequency). The top **600** per S1 go to stage 2.
- **Stage 2: re-rank.** rapidfuzz token-set similarity of name and address. A missing signal (Indic script, empty address) counts as neutral. We keep the **top 20 per S1 plus the top 3 S1s per record**: 26.6 per S1.
- **Stage 3: learned cascade filter.** The XGBoost pair model discards pairs with p ≤ 0.005. That is also the threshold below which pairs are never sent to the cross-encoders, so they can never be selected. What remains is the candidate set: **4.96 per S1** (median 5, 99th percentile 11), 8.6M test pairs compared with about 1.7e10 possible within-country pairs. It is written exactly to `candidate_pairs.tsv`.
  - On 275k held-out S1s, the filter keeps 99.96% of the true pairs that blocking found, and F0.5 is unchanged.
- **Scaling.** The index is partitioned by country and processed in chunks of S1s. Over-frequent keys are dropped by per-type caps, so each S1 touches a bounded number of postings and the work grows linearly with the data.
- **Pair recall on train** after blocking: US 98.5%, India 98.1% (95.4% / 93.7% of clusters kept complete). After the cascade: US 98.5%, India 98.0%. Raising the stage-1 pool from 150 to 600 was the largest single recall gain (+1.45 points for India).

---

## 4. Matching Model
- **Features (about 75):**
  - Name: token-set, sort and partial ratios, Jaro-Winkler on the no-space name, alias and domain flags, transliterated similarity.
  - Address: similarities; house-number equal, conflicting or truncated; compound numbers and units from the raw address.
  - Legal-form sets: added, dropped or conflicting.
  - Twin competition: other same-name candidates at the S1's own number.
  - Context: rank, margins to competing S1s, genericness counts, source.
- **Stage 1: XGBoost** on CUDA. A blend of two variants (depth 9, and depth 11 with colsample 0.6), 3-fold GroupKFold by S1 on 50% of train S1s.
- **Stage 2: cross-encoders** on the 0.005 < p < 0.995 pairs, input "name ; address" [SEP] "name ; address":

  | Model | Size | Held-out logloss |
  |---|---|---|
  | distilbert-multilingual (2 runs) | 134M | 0.205 / 0.167 |
  | xlm-roberta-base (1 and 2 epochs) | 278M | 0.162 / 0.157 |
  | mdeberta-v3-base (2 runs) | 278M | 0.155 / 0.152 |
  | xlm-roberta-large | 560M | 0.155 |
  | deberta-v3-large | 435M | 0.150 |

- **Stacker.** XGBoost on the GBM logit, the encoder logits and the encoder context in the S1 (rank, best other candidate, best competing owner). It is 2-fold cross-fitted on 275k S1s that none of the models trained on.
  - House-number, twin and legal-form features are **excluded**: their train-calibrated effect shifts on test.
  - France (final v16) uses a base stack of the multilingual, France-adapted encoders, with a +1 logit push: v1, v2, XLM-R-base-fr, and the round-2 XLM-R-base, mDeBERTa-v3-base and XLM-R-large (see below).
  - For the generator's filler variants ("X SARL Développement" for "X Santé SARL", same number and street, no S1 carrying the record's name), France takes the higher p of that stack and of the same stack built with the France-adapted DistilBERT v2.
  - The English deberta is excluded for France. It rejects these variants, and their US analogue is 99.5% true.
- **Round-2 France adaptation.**
  - **Labels:** the final France predictions of the validated recipe. Co-located noun swaps are negatives, and filler variants are positives.
  - **Synthetic negatives:** only pure sibling qualifiers and legal forms at a nearby number.
  - **Data:** 600k original pairs plus 388k France pairs.
  - **Models:**

    | Model | Where it ran | Held-out logloss |
    |---|---|---|
    | XLM-R-large, full train plus France | Modal H100 | 0.1549, equal to the original |
    | XLM-R-base, continued from its epoch-2 checkpoint | Kaggle T4 | 0.1605 |
    | mDeBERTa-v3-base | Modal H100 | 0.1649 |

  - The France base stack with these models scores 0.98805 / 0.98779 held-out (India / US).
  - On France, the models mainly recover acronyms, invented brand names and filler variants at the S1's exact address.
- **France co-located sibling rule.**
  - **Pattern:** same number and street; the names differ by one S1-vocabulary noun ("Gospel Sportive SARL" / "Gospel Club SARL").
  - **Why we treat them as separate businesses:** for 43–66% of such records another S1 carries exactly the record's name at that number. For filler words the figure is 0–15%. The number of such swaps per noun scales with that noun's frequency in S1 names.
  - **Scale:** there are 265 per 1k France S1s, against about 2 per 1k (1% true) in US/India, so the models give them p ≈ 0.26.
  - **Action:** p is capped for these pairs. Held-out effect on US/India: −0.00005.
  - **Leaderboard, US/India rows unchanged:** 0.983929 → **0.985752** (+0.0122 France F0.5; nearly every removed match was false). The extended rule plus the filler step reached **0.986229**.
- **Selection.** Exclusive-owner probabilities, then a global one-owner assignment. Per S1, choose the prefix size m that maximises 1.25·Σp / (0.25·E|T| + m); predict empty when Π(1 − p) is higher.

---

## 5. Results & Error Analysis
- **Held-out macro F0.5**, full pipeline, 275k unseen S1s:

  | Stage | India | US |
  |---|---|---|
  | GBM | 0.9783 | 0.9820 |
  | + one encoder | 0.9863 | 0.9864 |
  | + three encoders | 0.9878 | 0.9876 |
  | + eight encoders | **0.9881** | **0.9879** |

- **Public leaderboard:**

  | Version | Score |
  |---|---|
  | v03 | 0.9497 |
  | v04 (legal-form, number and twin features) | 0.9622 |
  | v07 (pool 600, blend) | 0.9671 |
  | + 1 encoder | 0.9789 |
  | + 3 encoders | 0.9825 |
  | + 5 encoders, France-adapted | 0.9839 |
  | + France co-located sibling rule | 0.985752 |
  | + extended rule and French filler variants | 0.986229 |
  | + round-2 France adaptation (XLM-R-large) | 0.986586 |
  | + all round-2 models (v16, final) | **0.98674** |

  The France adaptation alone was worth about +0.008 France F0.5, measured through the France-empty probe.
- **Measured and rejected (no held-out gain):**
  - label-shift EM
  - veto rules
  - calibration tuning
  - an empty-address specialist
  - retrieval expansions (transitive siblings, brand-by-address)
  - a legal-form rule
  - cluster-size balancing
  - a synthetic-sibling encoder
  - a uniform France recall push (flat on the leaderboard)
- **Remaining errors:**
  - **Generic-name collisions with empty-address records in France.** 5–7 S1s share the same name and the record has no address. This is irreducible under F0.5: a record should be claimed only at q ≥ 0.77.
  - **S1 twins with the same name and address** (France 9.2% of S1s).
  - **Blocking misses** (about 1.5–2% of pairs).
  - **Alias records with random names.**

---

## 6. Conclusion
The largest gains came from two things:
- **Reading the raw text with cross-encoders**, but only where the GBM was uncertain. Together with the cascade, that keeps the candidate set to 5 per entity and the encoder inference to 3.5M pairs.
- **Making selection match the metric:** exclusive owners and expected F0.5.

For the unlabelled country, we measured the train→test shift instead of assuming it. A single diagnostic upload split the leaderboard score by country. Structural statistics (twin shares, frequency scaling, cluster-size signatures) then turned the generator's patterns into rule-compliant adaptations. Everything was trained on the provided data only.

---

## Appendix

### A. Code Artefacts
The code is in `code/business_entity_resolution/`, and `README.md` lists every command (steps 1–12). Main modules:

| Module | Role |
|---|---|
| `normalize`, `translit` | Normalisation and learned transliteration |
| `blocking`, `build_candidates` | Candidates |
| `features`, `extra_features` | Pair features |
| `train_gpu`, `ensemble` | XGBoost and blend |
| `ce_pairs`, `cross_encoder`, `modal_ce` | Encoder data, training and scoring |
| `ce_france`, `ce_france_r2` | France self-training sets (round 1, round 2) |
| `ce_stack` | Cross-fitted stacker |
| `swap_rules`, `france_final` | France sibling rule, filler variants, final assembly |
| `postprocess` | Exclusive owner, one-owner, expected F0.5 |
| `predict`, `submission` | Writing and the official validator |
| `score` | Exact metric |

### B. Additional Results
Every upload's version, sha1, held-out scores, leaderboard score and pre-registered leaderboard prediction are in `submissions/submission_log.csv`.
