"""Free the focused exploration worker only after its first resumable checkpoint."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/replay_optimization_v3'
FOCUS=RUN/'focus_replay_320'


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def main():
    lock=json.loads((RUN/'candidate_lock.json').read_text(encoding='utf-8'))
    if not lock.get('selection_closed') or lock['candidate']!='guided_foreground_specialist_v1':
        raise ValueError('Close selection before deferring the secondary experiment')
    print(json.dumps(dict(stage='waiting_for_first_focus_checkpoint',goal_still_active=True)),flush=True)
    deadline=time.monotonic()+3600
    while True:
        state=json.loads((FOCUS/'status.json').read_text(encoding='utf-8'))
        epoch=state.get('completed_epochs',0)
        if epoch>=1 and state['stage']=='milestone_evaluation':
            break
        if state['stage'] in {'failed','training_complete','deferred_after_checkpoint'}:
            print(json.dumps(dict(stage='no_defer_needed',training_state=state['stage'])),flush=True);return
        result=RUN/'sealed_test_result.json'
        if result.exists() and not json.loads(result.read_text(encoding='utf-8'))['passes_joint_test_gate']:
            print(json.dumps(dict(stage='keep_exploring_after_failed_test')),flush=True);return
        if time.monotonic()>deadline:
            raise TimeoutError('No checkpoint boundary reached; training left untouched')
        time.sleep(5)
    checkpoint=FOCUS/'segment/weights'/f'epoch{epoch-1}.pt'
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError('Focused checkpoint is incomplete')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    immutable=RUN/'candidates'/state['candidate']/'model.pt'
    if hashlib.sha256(immutable.read_bytes()).hexdigest()!=sha:
        raise ValueError('Immutable milestone was not completely copied')
    targets=[]
    for process in psutil.process_iter(['pid','name']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            command=process.cmdline()
            if command and Path(command[0]).resolve().is_relative_to(ROOT/'.venv-vision') and any(Path(arg).name=='train_focus_replay.py' for arg in command[1:]):
                targets.append(process)
        except (psutil.AccessDenied,psutil.NoSuchProcess):
            continue
    for process in targets:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _,alive=psutil.wait_procs(targets,timeout=10)
    if alive:
        raise RuntimeError('Focused worker did not exit')
    reason='A complete recipe was already locked; preserve exploration and prioritize its independent test'
    save(FOCUS/'deferred.json',dict(at=datetime.now(timezone.utc).isoformat(),prior_status=state,reason=reason,
          checkpoint=checkpoint.relative_to(ROOT).as_posix(),sha256=sha,pids=[p.pid for p in targets],goal_still_active=True))
    save(FOCUS/'status.json',{**state,'stage':'deferred_after_checkpoint','reason':reason,'resume_checkpoint':checkpoint.relative_to(ROOT).as_posix()})
    print(json.dumps(dict(stage='deferred_after_checkpoint',checkpoint=str(checkpoint),sha256=sha,goal_still_active=True)),flush=True)


if __name__=='__main__':
    main()
