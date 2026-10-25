"""Run the whole analysis in order (about 40-50 min on a 2-core laptop, less with more cores).

    python run/run_all.py                 # full analysis
    WASPS_QUICK=1 python run/run_all.py   # smoke test with tiny numbers of repetitions (~2 min)
    WASPS_JOBS=4 python run/run_all.py    # limit the number of parallel processes

Every step writes its figures to paper/figures/, its tables and numbers to paper/generated/, and a
JSON/CSV copy to data/results/. Commit and push, and the paper in Overleaf shows the new values.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_community_description.py", "02_classifier_benchmark.py",
         "03_homogenization_test.py", "04_species_interpretation.py", "05_sensitivity.py"]

for step in STEPS:
    t0 = time.time()
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True)
    print(f"----- {step}: {time.time() - t0:.0f} s", flush=True)
