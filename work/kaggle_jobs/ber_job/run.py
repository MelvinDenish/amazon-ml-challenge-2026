"""Kaggle scripted job for the Bedrock ER pipeline (pushed via `kaggle kernels push`).

Installs the three pip dependencies, finds kaggle_runner.py in the attached
private dataset (melvindenishl/amazon-ml) and runs one pipeline job. Output
files land in /kaggle/working/artifacts and are pulled with `kaggle kernels output`.
"""
import glob
import subprocess
import sys

JOB_ARGS = ["--job", "cands", "--split", "train", "--countries", "India"]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "polars", "xgboost"], check=True)
runner = glob.glob("/kaggle/input/**/src/kaggle_runner.py", recursive=True)[0]
print("runner:", runner, "args:", JOB_ARGS, flush=True)
subprocess.run([sys.executable, runner, *JOB_ARGS], check=True)
