"""Run git operations to clean up tracked heavy files and commit."""
import subprocess, sys, os

ROOT = r"c:\Users\User\Documents\4rth semester\AI\SisFall_dataset"
os.chdir(ROOT)

def git(args, check=False):
    result = subprocess.run(
        ["git"] + args,
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("ERR:", result.stderr.strip())
    return result.returncode

# 1. Remove cached heavy / temp files (errors are OK if they weren't tracked)
to_untrack = [
    "submission/.feature_cache",
    "submission/__pycache__",
    "submission/src/__pycache__",
    "submission/results/*.pkl",
    "submission/splice_main.py",
    "submission/debug_runner.py",
    "submission/run_pipeline.py",
    "submission/_new_main_block.py",
    "submission/debug_log.txt",
    "submission/import_check.txt",
    "submission/dir_check.txt",
    "submission/check.txt",
    ".venv",
]
for path in to_untrack:
    git(["rm", "-r", "--cached", "--ignore-unmatch", path])

# 2. Stage .gitignore updates + all source changes
git(["add", ".gitignore", "submission/.gitignore"])
git(["add", "submission/main.py"])
git(["add", "submission/src/"])
git(["add", "submission/README.md"])
git(["add", "submission/results/"])   # txt files only (pkl already gitignored)
git(["status", "--short"])

# 3. Commit
rc = git(["commit", "-m",
    "Add multi-model LOSO pipeline + gitignore heavy files\n\n"
    "- main.py: comprehensive 11-phase pipeline (RF/XGB/MLP/SVM LOSO,\n"
    "  3-way subject split, Young->Elderly transfer, feature analysis,\n"
    "  SHAP explainability, sealed-test final evaluation)\n"
    "- src/model.py: add FallDetectorSVM, FallDetectorLDA\n"
    "- .gitignore: exclude dataset, .feature_cache, *.pkl, __pycache__,\n"
    "  venv, and temp debug scripts"
])
print("commit rc:", rc)
