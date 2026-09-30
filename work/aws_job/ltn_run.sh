#!/bin/bash
# Lightning job: XLM-R-large cross-encoder -> scores -> upload to S3 via presigned PUT URLs.
set -e
cd /teamspace/studios/this_studio
P=/home/zeus/miniconda3/envs/cloudspace/bin/python
TAG=$1; MODEL=$2; DATA=$3; BS=${4:-64}; LR=${5:-1e-5}
mkdir -p bedrock/out_$TAG
nvidia-smi --query-gpu=name,memory.total --format=csv
SM_CHANNEL_DATA=bedrock/$DATA SM_MODEL_DIR=bedrock/out_$TAG $P train_ce.py --model $MODEL --bs $BS --lr $LR ${EXTRA} 2>&1 | tee bedrock/out_$TAG/log.txt
$P - <<PY
import json, requests
u = json.load(open("urls_$TAG.json"))["put"]
for k in ("eval", "eval_x", "test_x", "test"):
    requests.put(u[k], data=open(f"bedrock/out_$TAG/{k}_scores.parquet", "rb").read(), timeout=600).raise_for_status()
    print("uploaded", k)
PY
