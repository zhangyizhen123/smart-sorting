"""Run the locked second-round comparison once, with a durable UTF-8 log."""
import ctypes
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from evaluate_replay_optimization import write_json

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/segment/joint_round_v4'


def status(stage,**fields):
    value=dict(stage=stage,at=datetime.now(timezone.utc).isoformat(),**fields)
    write_json(RUN/'sealed_driver_status.json',value);print(json.dumps(value),flush=True)


def main():
    script=ROOT/'vision/scripts/sealed_joint_round_test.py'
    with (RUN/'sealed_run.log').open('a',encoding='utf-8') as log:
        status('locking_second_round_candidate')
        subprocess.run([sys.executable,'-u',str(script),'lock'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        lock=json.loads((RUN/'candidate_lock.json').read_text())
        status('evaluating_locked_second_round',candidate=lock['candidate'],pipeline_sha256=lock['pipeline_sha256'])
        child=subprocess.Popen([sys.executable,'-u',str(script),'evaluate'],cwd=ROOT,stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,encoding='utf-8',bufsize=1)
        for line in child.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        if child.wait()!=0: raise RuntimeError('Second-round evaluator failed; see sealed_run.log')
    result=json.loads((RUN/'sealed_test_result.json').read_text())
    status('sealed_second_round_complete',passes_joint_test_gate=result['passes_joint_test_gate'],automatic_deployment=False)


if __name__=='__main__':
    try:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        main()
    except Exception as error:
        status('failed',error=str(error),automatic_deployment=False);raise
    finally:
        if hasattr(ctypes,'windll'): ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
