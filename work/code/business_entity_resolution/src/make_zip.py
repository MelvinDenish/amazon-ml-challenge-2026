"""Package a submitted version into the final-submission zip layout.

    python make_zip.py --version v02_xgb --team <team_name>

Creates work/submissions/<version>/<team>_submission.zip containing
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/{src/*.py, README.md, requirements.txt}
    Documentation_template.md        (the filled methodology write-up)
and a plain copy of the code in work/submissions/<version>/code_snapshot/ so
every leaderboard upload keeps its exact source (version history).
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from submission import SUBMISSION_DIR

CODE_ROOT = Path(__file__).resolve().parents[1]  # .../business_entity_resolution
DOC_PATH = SUBMISSION_DIR.parent / "Documentation_template.md"


def code_files() -> list[Path]:
    """Source files shipped in the zip: all src/*.py plus README and requirements."""
    files = sorted((CODE_ROOT / "src").glob("*.py"))
    return files + [CODE_ROOT / "README.md", CODE_ROOT / "requirements.txt"]


def make_zip(version: str, team: str) -> Path:
    """Write the code snapshot and the submission zip for ``version``."""
    vdir = SUBMISSION_DIR / version
    outputs = [vdir / "matching_results.tsv", vdir / "candidate_pairs.tsv"]
    missing = [p for p in outputs + [DOC_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")

    snap = vdir / "code_snapshot"
    if snap.exists():
        shutil.rmtree(snap)
    for f in code_files():
        dest = snap / f.relative_to(CODE_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

    zpath = vdir / f"{team}_submission.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in outputs:
            z.write(p, f"output/{p.name}")
        for f in code_files():
            z.write(f, f"code/business_entity_resolution/{f.relative_to(CODE_ROOT).as_posix()}")
        z.write(DOC_PATH, "Documentation_template.md")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB)")
    return zpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--team", default="team")
    a = ap.parse_args()
    make_zip(a.version, a.team)
