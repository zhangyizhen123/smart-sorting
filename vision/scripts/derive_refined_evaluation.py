"""Evaluate a classifier-refined candidate from cached native masks, without new detection."""
import copy,json,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from ultralytics.engine.results import Masks
from shape_refiner import PairwiseRefiner
from evaluate_replay_optimization import summarize,write_json,digest,ROOT


def derive(source,name,refiner_path,dataset):
    source=Path(source);out=source.parent/name
    if json.loads((source/'status.json').read_text(encoding='utf-8'))['stage']!='complete':raise ValueError('Raw evaluation incomplete')
    original=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    freeze={'source_freeze_sha256':digest(source/'freeze.json'),'source_summary_sha256':digest(source/'summary.json'),
            'model_sha256':original['provenance']['model_sha256'],'settings':original['provenance']['settings'],
            'refiner_sha256':digest(refiner_path/'refiner.pt'),'refiner_metadata_sha256':digest(refiner_path/'metadata.json'),
            'derivation_sha256':digest(Path(__file__)),'geometry_unchanged':True,'role':'candidate_refined'}
    if (out/'freeze.json').exists():
        if json.loads((out/'freeze.json').read_text(encoding='utf-8'))!=freeze:raise ValueError('Derived evaluation changed')
    else:write_json(out/'freeze.json',freeze)
    refiner=PairwiseRefiner(refiner_path);records=[]
    for path in sorted((source/'records').glob('*.json')):
        target=out/'records'/path.name
        if target.exists():
            record=json.loads(target.read_text(encoding='utf-8'))
            for asset,sha in record['mask_hashes'].items():
                if digest(out/asset)!=sha:raise ValueError('Derived mask changed')
        else:
            record=json.loads(path.read_text(encoding='utf-8'))
            with Image.open(dataset/record['image']) as image:array=np.asarray(image.convert('RGB'))
            tick=time.perf_counter();predictions=record['predictions']
            if predictions:
                arrays=[]
                for index in range(len(predictions)):
                    mask_file=source/'masks'/record['key']/f'{index+1:02d}.png'
                    arrays.append((np.asarray(Image.open(mask_file))>0).astype(np.uint8))
                polygons=Masks(torch.from_numpy(np.stack(arrays)),(record['height'],record['width'])).xy
                for item,polygon in zip(predictions,polygons,strict=True):
                    base=item['base_class_id'];item['class_id']=int(refiner.refine(array,polygon,base,item['confidence']))
            record['refiner_seconds']=time.perf_counter()-tick
            record['inference_seconds']+=record['refiner_seconds']
            for asset,sha in record['mask_hashes'].items():
                if digest(source/asset)!=sha:raise ValueError('Source mask changed')
                destination=out/asset;destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists():
                    if digest(destination)!=sha:raise ValueError('Existing mask mismatch')
                else:shutil.copy2(source/asset,destination)
            write_json(target,record)
        records.append(record)
    groups={'synthetic_validation':[r for r in records if r['domain']!='real'],
            'positive_validation':[r for r in records if r['domain'] in {'legacy','focused'}],
            **{domain:[r for r in records if r['domain']==domain] for domain in ['legacy','focused','negative','real']},
            'real_empty':[r for r in records if r['category']=='empty_tray'],
            'real_white':[r for r in records if r['category']=='white_cube']}
    summary={'name':name,'provenance':freeze,'image_count':len(records),'validation_images':1108,'real_train_fit_images':46,
             'real_independent_test_images':0,'test_used_for_selection':False,'groups':{key:summarize(values) for key,values in groups.items()}}
    for key in ['negative','real_empty']:
        group=groups[key];summary['groups'][key]['empty_false_detection_images']=sum(bool(r['predictions']) for r in group)
        summary['groups'][key]['empty_false_detection_rate']=sum(bool(r['predictions']) for r in group)/len(group)
    write_json(out/'summary.json',summary);write_json(out/'status.json',{'stage':'complete','images':len(records),'cached_geometry_reused':True})
    return summary
