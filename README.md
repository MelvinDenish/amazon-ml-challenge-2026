# Amazon ML Challenge 2026 — Business Entity Resolution (Team Bedrock)

Final public leaderboard: **0.98674** (v16). Top-100 cutoff was >= 0.99; see the post-mortem below.

## What is here (code only)
- `work/code/business_entity_resolution/` — the submitted pipeline (`src/`, `README.md` with steps 1–13, pinned `requirements.txt`)
- `work/analysis/` — every analysis/probe script and its log
- `work/kaggle_jobs/`, `work/aws_job/` — Kaggle kernels and Modal/SageMaker GPU job scripts
- `work/Documentation_template.md` — methodology write-up; `work/submissions/submission_log.csv` — every upload, sha1, LB score
- `docs/`, `student_resource/` — competition docs and utilities (dataset excluded)

## Full backup (data, artifacts, submissions, final zip) — 59 GB
Stored in AWS S3 (account 987569577835, CLI profile `old`), storage class Glacier Instant Retrieval:

```
s3://bedrock-er-987569577835-aps1/amazon_ml_backup/
```

Restore everything (≈59 GB download, ~$8 in retrieval + transfer fees):
```
aws s3 sync s3://bedrock-er-987569577835-aps1/amazon_ml_backup/ ./Amazon_ML --profile old --region ap-south-1
```
Restore just the final submission zip:
```
aws s3 cp s3://bedrock-er-987569577835-aps1/amazon_ml_backup/work/submissions/v16_r2_all/Bedrock_submission.zip . --profile old --region ap-south-1
```

## Post-mortem (why 0.98674 and not >= 0.99)
Measured on 275k held-out S1s: blocking missed 1.5–2% of true pairs (−0.005 to −0.006 F0.5), and the model missed hard
positives (−0.006 to −0.007). A perfect classifier on our candidates would have scored 0.994–0.995. The strongest model
family (cross-encoders) arrived at hour 34, GPU compute came on the last night, and the final 36 hours went to small
France rules instead of recall and model capacity.
