"""Kaggle job (Bedrock ER): rebuild candidates with blocking v3 (pool 600) for one split.

Inputs: private datasets melvindenishl/amazon-ml (data/*.parquet) and
melvindenishl/bedrock-src (src/*.py, translit_map.parquet). The code is taken
ONLY from bedrock-src (amazon-ml also contains an old src/ copy).
Outputs: /kaggle/working/artifacts/cands/<split>_<country>.parquet.
"""
import glob
import os
import shutil
import subprocess
import sys

SPLIT = "test"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz==3.14.6", "polars==1.44.2"], check=True)
bc = [p for p in glob.glob("/kaggle/input/**/build_candidates.py", recursive=True) if "bedrock-src" in p]
code = os.path.dirname(bc[0])
src_text = open(bc[0], encoding="utf-8").read()
assert '"pool_k": 600' in src_text, "stale code: pool_k is not 600"
data = os.path.dirname(glob.glob("/kaggle/input/**/amazon-ml/**/train_source1.parquet", recursive=True)[0])
tmap = [p for p in glob.glob("/kaggle/input/**/translit_map.parquet", recursive=True) if "bedrock-src" in p]
work = "/kaggle/working/artifacts"
os.makedirs(f"{work}/data", exist_ok=True)
for f in glob.glob(f"{data}/*.parquet"):
    dst = f"{work}/data/{os.path.basename(f)}"
    if not os.path.exists(dst):
        os.symlink(f, dst)
shutil.copy(tmap[0], f"{work}/translit_map.parquet")
shutil.copytree(code, "/kaggle/working/src", dirs_exist_ok=True)
print("code:", code, "| data:", data, "| map:", tmap[0], "| split:", SPLIT, flush=True)
env = {**os.environ, "BER_ARTIFACT_DIR": work}
subprocess.run([sys.executable, "-X", "utf8", "build_candidates.py", "--split", SPLIT],
               cwd="/kaggle/working/src", env=env, check=True)
shutil.rmtree(f"{work}/data")
os.remove(f"{work}/translit_map.parquet")
