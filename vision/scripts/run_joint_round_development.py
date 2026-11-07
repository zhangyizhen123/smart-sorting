"""Wait for development caches, select and verify; never open the final test."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
from evaluate_replay_optimization import write_json

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/segment/joint_round_v4'


def main():
    while True:
        path=RUN/'development/status.json'
        if path.exists() and json.loads(path.read_text())['stage']=='complete': break
        time.sleep(20)
    subprocess.run([sys.executable,'-u','vision/scripts/select_joint_round_candidate.py'],cwd=ROOT,check=True)
    if json.loads((RUN/'selection.json').read_text())['selected'] is None:
        write_json(RUN/'driver_status.json',dict(stage='no_development_candidate',test_exposed=False));return
    subprocess.run([sys.executable,'-u','vision/scripts/bundle_joint_round_candidate.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-u','vision/tests/check_joint_round_runtime.py'],cwd=ROOT,check=True)
    write_json(RUN/'driver_status.json',dict(stage='ready_for_candidate_lock',test_exposed=False))


if __name__=='__main__':
    try:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        main()
    except Exception as error:
        write_json(RUN/'driver_status.json',dict(stage='failed',error=str(error),test_exposed=False));raise
    finally:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
