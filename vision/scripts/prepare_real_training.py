"""Convert explicitly approved masks, then freeze a mixed pilot without real-data leakage."""
from __future__ import annotations
import csv
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import cv2
import numpy as np
from PIL import Image
import yaml
from import_real_review import NAMES, contained, digest, validate_review

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'vision/configs/real_pilot_v1.json'


def mask_polygon(mask):
    binary=(mask>0).astype('uint8')
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)!=1 or hierarchy is None:
        raise ValueError('YOLO export requires one connected instance without holes; do not bridge or fill it silently')
    polygon=contours[0].reshape(-1,2)
    if len(polygon)<3 or cv2.contourArea(polygon)<20:
        raise ValueError('Degenerate visible mask')
    h,w=binary.shape
    normalized=polygon.astype(float)/np.array([w,h])
    # Verify the serialized normalized coordinates, including round-trip rounding.
    normalized=np.array([float(f'{v:.8f}') for v in normalized.ravel()]).reshape(-1,2)
    reconstructed=np.zeros_like(binary)
    cv2.fillPoly(reconstructed,[np.rint(normalized*[w,h]).astype('int32')],1)
    iou=np.count_nonzero(reconstructed&binary)/max(1,np.count_nonzero(reconstructed|binary))
    if iou<.995:raise ValueError(f'Polygon conversion loses visible geometry: IoU {iou}')
    return normalized,iou


def balanced_legacy(rows,count,seed):
    rng=random.Random(seed);pool=list(rows);rng.shuffle(pool)
    selected=[];coverage=Counter()
    for turn in range(count):
        target=min(range(9),key=lambda cls:(coverage[cls],cls))
        candidates=[r for r in pool if target in r['classes']]
        if not candidates:raise ValueError('Cannot retain nine-class legacy coverage')
        row=candidates[0];pool.remove(row);selected.append(row);coverage.update(row['classes'])
    return selected


def verify(root):
    lock=json.loads((root/'freeze.json').read_text(encoding='utf-8'))
    if digest(root/'manifest.jsonl')!=lock['manifest_sha256']:raise ValueError('Mixed manifest changed')
    records=[json.loads(line) for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    hashes={};groups={};counts=Counter()
    for row in records:
        split=row['split'];counts[f'{row["domain"]}_{split}']+=1
        for key in ['image','label']:
            if digest(contained(root,row[key]))!=row[key+'_sha256']:raise ValueError(f'Changed {key}: {row[key]}')
        for key,seen in [(row['image_sha256'],hashes),(row['group_id'],groups)]:
            if key in seen and seen[key]!=split:raise ValueError('Cross-split image or group leakage')
            seen[key]=split
        if row['domain']=='real' and split!='train':raise ValueError('Shared physical real objects leaked into validation')
        if split not in {'train','val'}:raise ValueError('Pilot must not read a test split')
    if dict(counts)!=lock['counts']:raise ValueError('Frozen counts changed')
    return lock


def prepare():
    config=json.loads(CONFIG.read_text(encoding='utf-8'));out=ROOT/config['dataset']
    if (out/'freeze.json').exists():
        lock=verify(out)
        if lock['config_sha256']!=digest(CONFIG):raise ValueError('Pilot config changed; use a new dataset version')
        return lock
    if out.exists() and any(out.iterdir()):raise FileExistsError('Refusing to overwrite an unfinished/nonempty dataset')
    review=ROOT/config['review'];draft=json.loads((review/'annotations.json').read_text(encoding='utf-8'))
    pointer=json.loads((review/'current_review.json').read_text(encoding='utf-8'))
    snapshot=contained(review,pointer['snapshot'])
    if digest(snapshot)!=pointer['snapshot_sha256']:raise ValueError('Review snapshot changed')
    exported=json.loads(snapshot.read_text(encoding='utf-8'))['export'];approved=validate_review(draft,exported)
    if approved['reviewed_image_count']!=46:raise ValueError('All current real images must have explicit approval')
    prepared=[];conversion=[]
    for row in approved['images']:
        if digest(ROOT/row['source'])!=row['source_sha256']:raise ValueError('Source photograph changed')
        image=review/row['image']
        with Image.open(image) as im:width,height=im.size
        if [width,height]!=row['annotation_size']:raise ValueError('Oriented image dimensions disagree')
        lines=[]
        for instance in row['instances']:
            mask_path=review/instance['mask']
            if digest(mask_path)!=instance['mask_sha256'] or not instance['boundary_human_approved']:raise ValueError('Unapproved or changed mask')
            with Image.open(mask_path) as im:mask=np.asarray(im)
            polygon,iou=mask_polygon(mask)
            lines.append(str(instance['class_id'])+' '+' '.join(f'{v:.8f}' for v in polygon.ravel()))
            conversion.append({'instance_id':instance['instance_id'],'mask_sha256':instance['mask_sha256'],'roundtrip_iou':iou,'vertices':len(polygon)})
        prepared.append({'domain':'real','split':'train','key':'real_'+row['image_id'],'source_image':image,
            'label_text':'\n'.join(lines)+('\n' if lines else ''),'classes':[i['class_id'] for i in row['instances']],
            'group_id':row['leakage_group'],'source_photo':row['source'],'source_photo_sha256':row['source_sha256'],
            'source_review_snapshot_sha256':pointer['snapshot_sha256'],'category':row['category'],
            'annotation_approved':True,'final_test_eligible':False})
    legacy=ROOT/config['legacy_dataset']
    with (legacy/'manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for row in rows:row['classes']=[int(v) for v in row['classes'].split(',') if v]
    selected=balanced_legacy([r for r in rows if r['split']=='train'],45,config['seed'])+[r for r in rows if r['split']=='val']
    for row in selected:
        prepared.append({'domain':'legacy','split':row['split'],'key':'legacy_'+Path(row['image']).stem,
            'source_image':legacy/row['image'],'source_label':legacy/row['label'],'classes':row['classes'],
            'group_id':'legacy_'+row['group_id'],'source_original_split':row['split']})
    focused=ROOT/config['focused_dataset'];frozen=json.loads((focused/'freeze.json').read_text(encoding='utf-8'))
    if digest(focused/'manifest.jsonl')!=frozen['manifest_sha256']:raise ValueError('Focused source manifest changed')
    focus_rows=[json.loads(line) for line in (focused/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    rng=random.Random(config['seed']);modes=['single','separated','multi','low_view','high_view','small','edge','low_contrast','contact','occlusion']
    for split,nmodes in [('train',7),('val',10)]:
        for c,name in enumerate(NAMES):
            chosen_modes=[modes[(c*3+j)%10] for j in range(nmodes)]
            for mode in chosen_modes:
                candidates=sorted([r for r in focus_rows if r['split']==split and r['target_class']==name and r['mode']==mode],key=lambda r:r['image'])
                row=rng.choice(candidates)
                for key in ['image','label']:
                    if digest(focused/row[key])!=row[key+'_sha256']:raise ValueError('Selected focused source changed')
                prepared.append({'domain':'focused','split':split,'key':row['group_id'],'source_image':focused/row['image'],
                    'source_label':focused/row['label'],'classes':row['classes'],'group_id':row['group_id'],
                    'source_original_split':split,'mode':mode,'target_class':name})
    counts=Counter(f'{r["domain"]}_{r["split"]}' for r in prepared)
    if dict(counts)!=config['sample_counts']:raise ValueError(f'Unexpected sample counts: {counts}')
    out.mkdir(parents=True)
    records=[]
    for row in prepared:
        destination=out/'images'/row['split']/(row['key']+'.jpg');destination.parent.mkdir(parents=True,exist_ok=True)
        label=out/'labels'/row['split']/(row['key']+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row['source_image'],destination)
        if 'source_label' in row:shutil.copy2(row['source_label'],label)
        else:label.write_text(row['label_text'],encoding='utf-8')
        record={k:v for k,v in row.items() if k not in {'source_image','source_label','label_text'}}
        record.update(image=destination.relative_to(out).as_posix(),label=label.relative_to(out).as_posix(),
            image_sha256=digest(destination),label_sha256=digest(label),source_image=row['source_image'].relative_to(ROOT).as_posix())
        records.append(record)
    manifest=out/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    train_total=sum(r['split']=='train' for r in records)
    summary={'version':1,'config_sha256':digest(CONFIG),'review_snapshot_sha256':pointer['snapshot_sha256'],
        'manifest_sha256':digest(manifest),'counts':dict(counts),'training_images':train_total,'validation_images':len(records)-train_total,
        'sampling_fraction':{d:counts[d+'_train']/train_total for d in ['legacy','focused','real']},
        'real_instances':len(conversion),'real_empty_images':sum(r.get('category')=='empty_tray' for r in records),
        'minimum_polygon_roundtrip_iou':min(r['roundtrip_iou'] for r in conversion),'conversion':conversion,
        'real_split_policy':'One shared physical-object group, all train. Real diagnostics are training-fit only.',
        'no_test_split':True,'eligible_for_final_accuracy_claim':False}
    (out/'freeze.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'data.yaml').write_text(yaml.safe_dump({'path':out.resolve().as_posix(),'train':'images/train','val':'images/val','nc':9,'names':NAMES},allow_unicode=True),encoding='utf-8')
    verify(out)
    return {k:v for k,v in summary.items() if k!='conversion'}


if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False,indent=2))
