"""Preserve a completed epoch and defer only this project's exploratory workers."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
import psutil

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/segment/replay_optimization_v3'


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)


def main():
    chosen='guided_foreground_specialist_v1'
    comparison=json.loads((RUN/'development_comparison.json').read_text(encoding='utf-8'))
    if not comparison['candidates'].get(chosen,{}).get('passes_development_gate'):
        raise ValueError('Preserve active exploration until a complete candidate passes')
    state=json.loads((RUN/'status.json').read_text(encoding='utf-8'))
    epoch=state.get('completed_epochs',0)
    if epoch<1:
        raise ValueError('No completed epoch to resume')
    checkpoint=RUN/'head_replay/weights'/f'epoch{epoch-1}.pt'
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError('Checkpoint archive is incomplete')
    fingerprint=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    stopped=[]
    for script in ['watch_optimization_candidates.py','train_replay_optimization.py']:
        targets=[]
        for process in psutil.process_iter(['pid','name','cmdline']):
            try:
                cmd=process.info['cmdline'] or []
                if 'python' not in (process.info['name'] or '').lower() or not cmd:
                    continue
                if not Path(cmd[0]).resolve().is_relative_to(ROOT/'.venv-vision'):
                    continue
                if not any(Path(value).name==script for value in cmd[1:]):
                    continue
                targets.append(process)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        for process in targets:
            process.terminate();stopped.append(dict(pid=process.pid,script=script))
        _,alive=psutil.wait_procs(targets,timeout=10)
        if alive:
            raise RuntimeError('Exploration worker did not exit; do not overwrite its status')
    reason='Prioritize immutable final recipe evaluation; resume exploration from saved checkpoint if needed'
    write(RUN/'exploration_deferred.json',dict(at=datetime.now(timezone.utc).isoformat(),reason=reason,selected_candidate=chosen,
          prior_training_status=state,checkpoint=checkpoint.relative_to(ROOT).as_posix(),checkpoint_sha256=fingerprint,stopped=stopped,
          focused_320_training_unchanged=True,goal_still_active=True))
    write(RUN/'status.json',{**state,'stage':'deferred_after_checkpoint','reason':reason,'resume_checkpoint':checkpoint.relative_to(ROOT).as_posix()})
    write(RUN/'watcher_status.json',dict(stage='deferred_for_sealed_selection',reason=reason,goal_still_active=True))
    print(json.dumps(dict(deferred=stopped,checkpoint=str(checkpoint),sha256=fingerprint,goal_still_active=True)),flush=True)


if __name__=='__main__':
    main()
