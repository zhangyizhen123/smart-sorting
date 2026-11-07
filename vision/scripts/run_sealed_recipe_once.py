"""Wait for exact native equivalence, then lock/evaluate once with temporary wakefulness."""
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/replay_optimization_v3'
CANDIDATE='guided_foreground_specialist_v1'


def status(stage,**fields):
    value=dict(stage=stage,at=datetime.now(timezone.utc).isoformat(),**fields)
    path=RUN/'sealed_driver_status.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def main():
    try:
        if hasattr(ctypes,'windll'):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        status('waiting_for_exact_runtime_check',candidate=CANDIDATE)
        audit=RUN/'runtime_recipes'/CANDIDATE/'runtime_check.json';deadline=time.monotonic()+1200
        while True:
            if audit.exists():
                result=json.loads(audit.read_text(encoding='utf-8'))
                if result.get('passed') and result.get('torch_runtime_threads')==7 and result.get('box_tolerance_pixels')==0 and result.get('confidence_tolerance')==0:
                    break
            if time.monotonic()>deadline:
                raise TimeoutError('Exact runtime check has not passed; test remains closed')
            time.sleep(5)
        executable=ROOT/'.venv-vision/Scripts/python.exe';script=ROOT/'vision/scripts/sealed_recipe_test.py'
        status('locking_selection',candidate=CANDIDATE)
        subprocess.run([str(executable),'-u',str(script),'lock'],cwd=ROOT,check=True)
        status('evaluating_immutable_recipe',candidate=CANDIDATE)
        subprocess.run([str(executable),'-u',str(script),'evaluate'],cwd=ROOT,check=True)
        report=json.loads((RUN/'sealed_test_result.json').read_text(encoding='utf-8'))
        status('sealed_evaluation_complete',passes_joint_test_gate=report['passes_joint_test_gate'],automatic_deployment=False)
    except Exception as error:
        status('failed',error=str(error),automatic_deployment=False)
        raise
    finally:
        if hasattr(ctypes,'windll'):
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
