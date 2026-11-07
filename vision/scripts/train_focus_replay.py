"""Second bounded replay experiment; evaluate saved milestones between epochs."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
from evaluate_replay_optimization import ROOT, digest, evaluate, write_json
from prepare_replay_optimization import verify
from derive_refined_evaluation import derive
from compare_optimization_candidates import compare


def main():
    config = ROOT/'vision/configs/focus_replay_320_v3.json'
    cfg = json.loads(config.read_text(encoding='utf-8'))
    out = ROOT/cfg['output'];run = ROOT/cfg['evaluation_output'];data = ROOT/cfg['dataset']
    verify(data)
    if (out/'segment').exists():
        raise FileExistsError('Training run already exists; inspect checkpoint instead of overwriting')
    source = ROOT/cfg['initial_model'];parameters = cfg['training'];model_hash = digest(source)
    torch_utils.NUM_THREADS = parameters['threads'];torch.set_num_threads(parameters['threads'])
    os.environ['OMP_NUM_THREADS'] = str(parameters['threads']);os.environ['MKL_NUM_THREADS'] = str(parameters['threads'])
    experiment = {'stage':'training', 'started_at':datetime.now(timezone.utc).isoformat(),
                  'config_sha256':digest(config), 'script_sha256':digest(Path(__file__)),
                  'initial_model_sha256':model_hash, 'parameters':parameters, 'deployment_changed':False}
    write_json(out/'experiment.json', experiment);write_json(out/'status.json', experiment)
    model = YOLO(str(source))

    def milestone(trainer):
        epoch = trainer.epoch+1
        state = {'stage':'training', 'completed_epochs':epoch, 'total_epochs':parameters['epochs'],
                 'metrics':{k:float(v) for k,v in trainer.metrics.items()}, 'updated_at':datetime.now(timezone.utc).isoformat()}
        write_json(out/'status.json', state)
        if epoch not in cfg['milestones']:
            return
        name = f'focus320_e{epoch:02d}'
        checkpoint = out/'segment/weights'/f'epoch{epoch-1}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError('Milestone checkpoint was not saved before evaluation')
        saved = run/'candidates'/name/'model.pt';saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.exists():
            if digest(saved) != digest(checkpoint):
                raise ValueError('Immutable candidate checkpoint changed')
        else:
            shutil.copy2(checkpoint, saved)
        spec = {'model':saved.relative_to(ROOT).as_posix(), 'role':'candidate',
                'settings':{'imgsz':320, 'conf':.25, 'iou':.7, 'device':'cpu', 'retina_masks':True, 'agnostic_nms':True}}
        write_json(out/'status.json', {**state, 'stage':'milestone_evaluation', 'candidate':name})
        evaluation_cfg = json.loads((ROOT/'vision/configs/replay_optimization_v3.json').read_text(encoding='utf-8'))
        if not (run/'evaluation'/name/'summary.json').exists():
            evaluate(name, spec, evaluation_cfg)
        if not (run/'evaluation'/(name+'_refined')/'summary.json').exists():
            derive(run/'evaluation'/name, name+'_refined', ROOT/'models/trained/stl_shape_focus_v2/refiner', data)
        comparison = compare(run)
        print(json.dumps({'milestone':epoch,'candidate':name,'selected_for_sealed_test':comparison.get('selected_for_sealed_test')}), flush=True)
        write_json(out/'status.json', state)

    model.add_callback('on_fit_epoch_end', milestone)
    try:
        model.train(data=str(data/'data.yaml'), project=str(out), name='segment',
                    epochs=parameters['epochs'], imgsz=parameters['imgsz'], batch=parameters['batch'],
                    device='cpu', workers=0, cache=False, seed=cfg['seed'], pretrained=True,
                    optimizer='AdamW', lr0=parameters['lr0'], lrf=parameters['lrf'], cos_lr=True,
                    warmup_epochs=1., patience=6, freeze=parameters['freeze'], amp=False,
                    mosaic=0., mixup=0., copy_paste=0., degrees=10., translate=.05, scale=.2,
                    shear=0., perspective=0., fliplr=.5, flipud=0., hsv_h=.05, hsv_s=.5, hsv_v=.35,
                    overlap_mask=True, mask_ratio=parameters['mask_ratio'], save=True, save_period=1,
                    plots=False, verbose=True)
        if digest(source) != model_hash:
            raise ValueError('Initial checkpoint changed during training')
        write_json(out/'status.json', {'stage':'training_complete','completed_at':datetime.now(timezone.utc).isoformat(),
                                      'deployment_changed':False,'test_used_for_selection':False})
    except Exception as exc:
        write_json(out/'status.json', {'stage':'failed','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
