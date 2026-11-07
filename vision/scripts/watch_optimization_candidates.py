"""Watch real training checkpoints, evaluate milestones, and preserve a joint gate.

Keeps Windows awake only while active work is being monitored; never promotes
a model or evaluates sealed test data automatically.
"""
import ctypes,json,shutil,time
from pathlib import Path
import psutil
import torch
from evaluate_replay_optimization import evaluate,write_json,digest,ROOT
from derive_refined_evaluation import derive
from compare_optimization_candidates import compare


def processes(script):
    matches=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            if 'python' in (p.info['name'] or '').lower() and any(script in part for part in (p.info['cmdline'] or [])):
                matches.append(p.info['pid'])
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    return matches


def main():
    cfg=json.loads((ROOT/'vision/configs/replay_optimization_v3.json').read_text(encoding='utf-8'))
    out=ROOT/cfg['output'];torch.set_num_threads(2)
    awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) if hasattr(ctypes,'windll') else None
    try:
        while True:
            control=out/'control.json'
            if control.exists() and json.loads(control.read_text(encoding='utf-8')).get('paused'):
                write_json(out/'watcher_status.json',{'stage':'paused_by_user'})
                break
            alive=processes('train_replay_optimization.py')
            status=json.loads((out/'status.json').read_text(encoding='utf-8'))
            completed=status.get('completed_epochs',0)
            if status['stage']=='training_complete':completed=cfg['training']['epochs']
            baseline_running=processes('evaluate_replay_optimization.py')
            baseline_complete=all((out/'evaluation'/name/'summary.json').exists() for name in ['v1_deployed','v2_deployed','pilot_640'])
            if not baseline_complete and not baseline_running:
                specs=json.loads((ROOT/'vision/configs/optimization_baselines_v3.json').read_text(encoding='utf-8'))
                for name,spec in specs.items():
                    if not (out/'evaluation'/name/'summary.json').exists():evaluate(name,spec,cfg)
                baseline_complete=True
            if baseline_complete:
                milestones=[epoch for epoch in [1,3,6] if epoch<=completed]
                for epoch in milestones:
                    source=out/'head_replay/weights'/f'epoch{epoch-1}.pt'
                    if not source.exists():continue
                    name=f'head_e{epoch:02d}_640'
                    sealed=out/'candidates'/name/'model.pt';sealed.parent.mkdir(parents=True,exist_ok=True)
                    if not sealed.exists():shutil.copy2(source,sealed)
                    elif digest(source)!=digest(sealed):raise ValueError('A milestone checkpoint changed')
                    spec={'model':sealed.relative_to(ROOT).as_posix(),'role':'candidate','settings':{'imgsz':640,'conf':.25,'iou':.7,'device':'cpu','retina_masks':True,'agnostic_nms':True}}
                    if not (out/'evaluation'/name/'summary.json').exists():
                        write_json(out/'watcher_status.json',{'stage':'candidate_evaluation','name':name,'training_pids':alive})
                        evaluate(name,spec,cfg)
                    refined=name+'_refined'
                    if not (out/'evaluation'/refined/'summary.json').exists():
                        derive(out/'evaluation'/name,refined,ROOT/'models/trained/stl_shape_focus_v2/refiner',ROOT/cfg['dataset'])
                    comparison=compare(out)
                    print(json.dumps({'milestone':epoch,'selected_for_sealed_test':comparison.get('selected_for_sealed_test')}),flush=True)
            if not alive:
                result=compare(out)
                write_json(out/'watcher_status.json',{'stage':'decision_needed','training_status':status['stage'],
                           'selected_for_sealed_test':result.get('selected_for_sealed_test'),
                           'note':'No automatic success or deployment; inspect all quality gates and sealed-test readiness.'})
                print(json.dumps({'stage':'decision_needed','training_status':status['stage']}),flush=True)
                break
            write_json(out/'watcher_status.json',{'stage':'waiting_for_verified_training','training_pids':alive,
                       'completed_epochs':completed,'baseline_processes':baseline_running,'awake_request_active':bool(awake)})
            time.sleep(20)
    finally:
        if hasattr(ctypes,'windll'):ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
