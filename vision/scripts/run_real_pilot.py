"""Paired local pilot: fixed mixed data, validation-only selection, real training-fit audit."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from evaluate_focused import score_records, digest, masks_from_truth
from shape_refiner import read_polygons
from prepare_real_training import verify

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'vision/configs/real_pilot_v1.json'


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def state(out,stage,**extra):
    record={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),**extra}
    write_json(out/'status.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


def evaluate(checkpoint,rows,dataset,settings,out):
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        existing=json.loads(out.read_text(encoding='utf-8'))
        if existing['model_sha256']!=digest(checkpoint) or existing['settings']!=settings:
            raise ValueError('Evaluation checkpoint or settings changed')
        return existing
    model=YOLO(str(checkpoint));records=[]
    for row in rows:
        path=dataset/row['image']
        with Image.open(path) as source:im=source.convert('RGB')
        truth=read_polygons(dataset/row['label'],im.width,im.height)
        result=model.predict(im,verbose=False,save=False,**settings)[0]
        preds=[];pred_masks=[]
        if result.boxes is not None and len(result.boxes):
            if result.masks is None:raise ValueError('Detector returned boxes without masks')
            for j,mask in enumerate(result.masks.data.cpu().numpy()):
                preds.append({'class_id':int(result.boxes.cls[j]),'confidence':float(result.boxes.conf[j]),
                              'box':result.boxes.xyxy[j].cpu().tolist()})
                pred_masks.append(cv2.resize(mask,(im.width,im.height),interpolation=cv2.INTER_NEAREST)>0.5)
        true_masks=masks_from_truth(truth,im.width,im.height)
        ious=np.zeros((len(truth),len(preds)),float)
        for a,tm in enumerate(true_masks):
            for b,pm in enumerate(pred_masks):ious[a,b]=np.count_nonzero(tm&pm)/max(1,np.count_nonzero(tm|pm))
        records.append({'image':row['image'],'domain':row['domain'],'category':row.get('category'),
            'truth_classes':[c for c,_ in truth],'predictions':preds,'ious':ious.tolist()})
    subsets=defaultdict(list)
    for r in records:subsets[r['domain']].append(r)
    report={'model':checkpoint.relative_to(ROOT).as_posix(),'model_sha256':digest(checkpoint),'settings':settings,
        'overall':score_records(records),'by_domain':{name:score_records(group) for name,group in subsets.items()},
        'real_training_fit_only':any(r['domain']=='real' for r in rows),'refiner_used':False}
    negatives=[r for r in records if r['category']=='empty_tray']
    report['empty_tray']={'images':len(negatives),'images_with_false_detections':sum(bool(r['predictions']) for r in negatives),
        'false_detections':sum(len(r['predictions']) for r in negatives)}
    white=[r for r in records if r['category']=='white_cube'];report['white_cube']=score_records(white) if white else None
    out.with_suffix('.records.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
    write_json(out,report)
    return report


def main():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():print((out/'summary.json').read_text(encoding='utf-8'));return
    data=ROOT/cfg['dataset'];lock=verify(data);train_cfg=cfg['training']
    if digest(CONFIG)!=lock['config_sha256']:raise ValueError('Training config no longer matches frozen data')
    originals={name:digest(ROOT/path) for name,path in cfg['initial_models'].items()}
    experiment={'config':cfg,'config_sha256':digest(CONFIG),'freeze_sha256':digest(data/'freeze.json'),
        'source_model_hashes':originals,'started_at':datetime.now(timezone.utc).isoformat(),
        'purpose':'Three-epoch CPU smoke experiment, not full convergence or an independent real accuracy test.'}
    if (out/'experiment.json').exists():
        previous=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
        for key in ['config_sha256','freeze_sha256','source_model_hashes']:
            if previous[key]!=experiment[key]:raise ValueError('Experiment changed; use a new output directory')
        experiment=previous
    else:write_json(out/'experiment.json',experiment)
    start=time.monotonic();torch.set_num_threads(train_cfg['threads'])
    import ultralytics.utils.torch_utils as torch_utils
    torch_utils.NUM_THREADS=train_cfg['threads']
    try:
        for name,initial in cfg['initial_models'].items():
            run=out/name;done=out/(name+'_training_complete.json')
            if done.exists():continue
            if run.exists():raise FileExistsError(f'Partial run exists: {run}; inspect before resuming')
            state(out,'training',candidate=name,epochs=train_cfg['epochs'],imgsz=train_cfg['imgsz'],log=(out/(name+'.log')).as_posix())
            command=[sys.executable,str(ROOT/'vision/scripts/train_seg.py'),'--data',str(data/'data.yaml'),'--model',initial,
                '--project',str(out),'--name',name,'--epochs',str(train_cfg['epochs']),'--imgsz',str(train_cfg['imgsz']),
                '--batch',str(train_cfg['batch']),'--device','cpu','--threads',str(train_cfg['threads']),
                '--workers','0','--cache','none','--focused','--no-amp','--seed',str(cfg['seed']),
                '--lr0',str(train_cfg['lr0']),'--patience',str(train_cfg['patience']),
                '--warmup-epochs',str(train_cfg['warmup_epochs'])]
            with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            for checkpoint in ['best.pt','last.pt']:
                if not (run/'weights'/checkpoint).exists():raise FileNotFoundError('Missing checkpoint')
            write_json(done,{'best_sha256':digest(run/'weights/best.pt'),'last_sha256':digest(run/'weights/last.pt'),
                'finished_at':datetime.now(timezone.utc).isoformat()})
        state(out,'validation')
        rows=[json.loads(line) for line in (data/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        validation=[r for r in rows if r['split']=='val'];real=[r for r in rows if r['domain']=='real']
        settings={k:v for k,v in cfg['evaluation'].items() if k!='match_iou'};settings['device']='cpu'
        checkpoints={**{name+'_initial':ROOT/path for name,path in cfg['initial_models'].items()},
                     **{name+'_'+kind:out/name/'weights'/f'{kind}.pt' for name in cfg['initial_models'] for kind in ['best','last']}}
        reports={}
        for name,checkpoint in checkpoints.items():
            reports[name]=evaluate(checkpoint,validation,data,settings,out/'evaluation'/f'val_{name}.json')
            scores={d:r['strict_instance_score'] for d,r in reports[name]['by_domain'].items()}
            print(json.dumps({'validation':name,'scores':scores}),flush=True)
        domains=['legacy','focused']
        baselines={d:max(reports[name+'_initial']['by_domain'][d]['strict_instance_score'] for name in cfg['initial_models']) for d in domains}
        eligible=[name for name in reports if not name.endswith('_initial') and all(reports[name]['by_domain'][d]['strict_instance_score']>=baselines[d] for d in domains)]
        ranking=sorted([name for name in reports if not name.endswith('_initial')],key=lambda name:sum(reports[name]['by_domain'][d]['strict_instance_score'] for d in domains),reverse=True)
        selected=next((name for name in ranking if name in eligible),None)
        diagnostic=selected or ranking[0]
        selection={'synthetic_nonregression_baselines':baselines,'eligible_candidates':eligible,'selected':selected,
                   'diagnostic_candidate':diagnostic,'selection_used_real_training_images':False,'selection_used_test':False,
                   'automatic_deployment':False,'real_independent_validation_available':False}
        write_json(out/'selection.json',selection)
        state(out,'real_training_fit_diagnostic',selection=selection)
        fits={}
        for name in ['from_v1_initial','from_v2_initial',diagnostic]:
            fits[name]=evaluate(checkpoints[name],real,data,settings,out/'evaluation'/f'real_train_fit_{name}.json')
        for name,path in cfg['initial_models'].items():
            if digest(ROOT/path)!=originals[name]:raise ValueError('Original deployed model changed')
        summary={'config_sha256':digest(CONFIG),'training_images':lock['training_images'],'validation_images':len(validation),
            'real_train_images':len(real),'real_independent_test_images':0,'epochs_per_origin':train_cfg['epochs'],'input_size':train_cfg['imgsz'],
            'selection':selection,'validation':{name:{d:{k:r[k] for k in ['instance_count','strict_instance_score','missed_instances','duplicate_detections','spurious_detections']} for d,r in report['by_domain'].items()} for name,report in reports.items()},
            'real_training_fit':{name:{'strict_instance_score':r['overall']['strict_instance_score'],'instances':r['overall']['instance_count'],
                'empty_tray':r['empty_tray'],'white_cube_strict_score':r['white_cube']['strict_instance_score']} for name,r in fits.items()},
            'elapsed_seconds':round((datetime.now(timezone.utc)-datetime.fromisoformat(experiment['started_at'])).total_seconds(),1),'cloud_spend':0,'models_deployed':False,
            'real_99_percent_target_verified':False,'completed_at':datetime.now(timezone.utc).isoformat()}
        write_json(out/'summary.json',summary);state(out,'complete',summary=(out/'summary.json').as_posix())
    except Exception as exc:
        state(out,'failed',error=str(exc));raise


if __name__=='__main__':main()
