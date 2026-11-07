"""One bounded CPU replay candidate; preserves all deployed weights."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import torch
from ultralytics import YOLO
import ultralytics.utils.torch_utils as torch_utils
from prepare_replay_optimization import verify

ROOT=Path(__file__).resolve().parents[2]


def write_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def main():
    config=ROOT/'vision/configs/replay_optimization_v3.json';cfg=json.loads(config.read_text(encoding='utf-8'))
    dataset=ROOT/cfg['dataset'];verify(dataset)
    root=ROOT/cfg['output'];root.mkdir(parents=True,exist_ok=True)
    if (root/'head_replay').exists():raise FileExistsError('Training directory already exists; inspect saved checkpoint before resume')
    parameters=cfg['training'];torch_utils.NUM_THREADS=parameters['threads'];torch.set_num_threads(parameters['threads'])
    os.environ['OMP_NUM_THREADS']=str(parameters['threads']);os.environ['MKL_NUM_THREADS']=str(parameters['threads'])
    model_path=ROOT/cfg['initial_model'];model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    record={'stage':'training','started_at':datetime.now(timezone.utc).isoformat(),'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
            'initial_model_sha256':model_sha,'parameters':parameters,'cloud_spend':0,'deployment_changed':False}
    write_json(root/'experiment.json',record);write_json(root/'status.json',record)
    model=YOLO(str(model_path))
    def progress(trainer):
        status={'stage':'training','completed_epochs':trainer.epoch+1,'total_epochs':trainer.epochs,
                'metrics':{key:float(value) for key,value in trainer.metrics.items()},'updated_at':datetime.now(timezone.utc).isoformat()}
        write_json(root/'status.json',status);print(json.dumps(status),flush=True)
    model.add_callback('on_fit_epoch_end',progress)
    try:
        model.train(data=str(dataset/'data.yaml'),project=str(root),name='head_replay',
                    epochs=parameters['epochs'],imgsz=parameters['imgsz'],batch=parameters['batch'],device='cpu',
                    workers=0,cache=False,seed=cfg['seed'],pretrained=True,optimizer='AdamW',lr0=parameters['lr0'],lrf=parameters['lrf'],
                    cos_lr=True,warmup_epochs=parameters['warmup_epochs'],patience=parameters['patience'],freeze=parameters['freeze'],
                    amp=False,mosaic=0.,mixup=0.,copy_paste=0.,degrees=10.,translate=.05,scale=.2,shear=0.,perspective=0.,
                    fliplr=.5,flipud=0.,hsv_h=.05,hsv_s=.5,hsv_v=.35,overlap_mask=True,mask_ratio=parameters['mask_ratio'],
                    save=True,save_period=1,plots=True,verbose=True)
        if hashlib.sha256(model_path.read_bytes()).hexdigest()!=model_sha:raise ValueError('Initial model changed')
        write_json(root/'status.json',{'stage':'training_complete','completed_at':datetime.now(timezone.utc).isoformat(),'deployment_changed':False})
    except Exception as exc:
        write_json(root/'status.json',{'stage':'failed','error':str(exc)});raise


if __name__=='__main__':main()
