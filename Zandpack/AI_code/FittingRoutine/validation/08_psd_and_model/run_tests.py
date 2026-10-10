import subprocess
import sys

ROOT = "/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine"
for script in [f"{ROOT}/tests/test_synthetic.py",
               f"{ROOT}/tests/test_model_form.py"]:
    print(f"\n===== {script} =====", flush=True)
    r = subprocess.run([sys.executable, "-u", script], cwd=ROOT,
                      capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    print("FAILED rc=%d" % r.returncode if r.returncode else "OK", flush=True)
    if r.returncode:
        print(r.stderr[-2000:], flush=True)
