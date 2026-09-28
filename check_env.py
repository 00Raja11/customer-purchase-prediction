"""
Environment & model sanity checker.
Run this BEFORE launching Streamlit. Fixes 99% of EOFError cases.
"""
import os
import sys
import joblib

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("Current folder    :", os.getcwd())

try:
    import sklearn, joblib as jl, streamlit
    print("sklearn version   :", sklearn.__version__)
    print("joblib  version   :", jl.__version__)
    print("streamlit version :", streamlit.__version__)
except Exception as e:
    print("❌ Import problem:", e)

print()
print("=" * 60)
print("MODEL FILES")
print("=" * 60)

MODELS_DIR = "models"
EXPECTED = [
    "logistic_model.pkl",
    "decision_tree_model.pkl",
    "random_forest_model.pkl",
    "knn_model.pkl",
    "scaler.pkl",
]

if not os.path.isdir(MODELS_DIR):
    print(f"❌ Folder '{MODELS_DIR}' does not exist.")
    sys.exit(1)

for fname in EXPECTED:
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print(f"❌ MISSING : {fname}")
        continue

    size = os.path.getsize(path)
    if size == 0:
        print(f"⚠️  EMPTY   : {fname} (0 bytes) — will cause EOFError")
        continue

    try:
        obj = joblib.load(path)
        print(f"✅ OK      : {fname:35s} {size:>10,} bytes → {type(obj).__name__}")
    except Exception as e:
        print(f"❌ CORRUPT : {fname:35s} {size:>10,} bytes → {type(e).__name__}: {e}")

print()
print("=" * 60)
print("If everything says ✅ → run: streamlit run app.py")
print("If ANY line shows ❌ or ⚠️ → retrain that model in the notebook.")
print("=" * 60)