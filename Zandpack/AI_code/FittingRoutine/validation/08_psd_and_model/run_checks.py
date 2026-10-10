import subprocess
import sys

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
for script in [
    "/tmp/nonuniform_test.py",
    "/tmp/nonuniform_moderate_test.py",
    f"{ROOT}/tests/test_synthetic.py",
    f"{ROOT}/tests/test_model_form.py",
]:
    print(f"\n===== {script} =====", flush=True)
    r = subprocess.run([sys.executable, "-u", script], cwd=ROOT,
                      capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-3000:], flush=True)
        print(f"FAILED rc={r.returncode}", flush=True)
    else:
        print("OK", flush=True)
