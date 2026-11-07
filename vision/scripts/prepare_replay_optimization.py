"""Freeze full legacy replay, balanced focused examples, real labels and negatives."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import yaml
from make_stl_synthetic import make_background
from predict_stl_demo import NAMES

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'vision/configs/replay_optimization_v3.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def negative_image(seed,size=640):
    """Known empty scene: no sorting objects are rendered; tray is background."""
    rng=random.Random(seed)
    background=make_background(size,rng).convert('RGB')
    if rng.random()<.2:return background,'empty_table'
    tray_size=400
    layer=Image.new('RGBA',(tray_size,tray_size),(0,0,0,0))
    draw=ImageDraw.Draw(layer)
    level=rng.randint(180,247);rim=rng.randint(8,16)
    draw.rounded_rectangle((4,4,395,395),radius=18,fill=(level-20,level-18,level-16,255))
    draw.rounded_rectangle((rim,rim,399-rim,399-rim),radius=12,fill=(level,level,level,255))
    hole_side=rng.choice([0,1,2,3]);cx=65 if hole_side in [0,3] else 335;cy=65 if hole_side in [0,1] else 335
    radius=rng.randint(19,32);draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=(0,0,0,0))
    src=np.float32([[0,0],[399,0],[399,399],[0,399]])
    margin=rng.uniform(55,125)
    dst=np.float32([[margin+rng.uniform(-25,30),margin+rng.uniform(-40,25)],
                    [size-margin+rng.uniform(-25,25),margin+rng.uniform(-20,50)],
                    [size-margin+rng.uniform(-45,25),size-margin+rng.uniform(-20,25)],
                    [margin+rng.uniform(-25,40),size-margin+rng.uniform(-40,20)]])
    matrix=cv2.getPerspectiveTransform(src,dst)
    warped=cv2.warpPerspective(np.asarray(layer),matrix,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    image=Image.alpha_composite(background.convert('RGBA'),Image.fromarray(warped)).convert('RGB')
    if rng.random()<.4:image=image.filter(ImageFilter.GaussianBlur(rng.uniform(.3,1.1)))
    return image,'empty_tray_procedural'


def copy_checked(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target):raise ValueError('Existing data mismatch: '+str(target))
    else:shutil.copy2(source,target)


def prepare():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['dataset']
    if (out/'freeze.json').exists():return verify(out)
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'build_inputs.json';inputs={'config_sha256':sha(CONFIG),'generator_sha256':sha(Path(__file__))}
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=inputs:raise ValueError('Partial dataset belongs to different inputs')
    marker.write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    rows=[]
    def add(source,source_image,source_label,domain,split,key,group,**extra):
        image=source/source_image.replace('\\','/');label=source/source_label.replace('\\','/')
        dest_image=out/'images'/split/(key+'.jpg');dest_label=out/'labels'/split/(key+'.txt')
        copy_checked(image,dest_image);copy_checked(label,dest_label)
        classes=[int(line.split()[0]) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        rows.append({'key':key,'domain':domain,'split':split,'group_id':group,'image':dest_image.relative_to(out).as_posix(),
                     'label':dest_label.relative_to(out).as_posix(),'image_sha256':sha(dest_image),'label_sha256':sha(dest_label),
                     'classes':classes,'source_image':image.relative_to(ROOT).as_posix(),**extra})
    legacy=ROOT/'dataset/stl_synthetic_yolo'
    with (legacy/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] in {'train','val'}:
                key='legacy_'+Path(r['image'].replace('\\','/')).stem
                add(legacy,r['image'],r['label'],'legacy',r['split'],key,'legacy_'+r['group_id'])
    focus=ROOT/'dataset/focused_v2';buckets=defaultdict(list)
    for line in (focus/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['split']=='train':buckets[(r['target_class'],r['mode'])].append(r)
        elif r['split']=='val':add(focus,r['image'],r['label'],'focused','val','focused_'+Path(r['image']).stem,r['group_id'],mode=r['mode'])
    if len(buckets)!=90:raise ValueError('Expected 9 classes x 10 training modes')
    rng=random.Random(cfg['seed'])
    for key,bucket in sorted(buckets.items()):
        selected=rng.sample(sorted(bucket,key=lambda r:r['image']),cfg['focused_train_per_class_mode'])
        for r in selected:add(focus,r['image'],r['label'],'focused','train','focused_'+Path(r['image']).stem,r['group_id'],mode=r['mode'])
    real=ROOT/'dataset/real_mixed_pilot_v1'
    for line in (real/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['domain']=='real':
            for repetition in range(cfg['real_repetitions']):
                add(real,r['image'],r['label'],'real','train',r['key']+f'_r{repetition}',r['group_id'],
                    category=r['category'],repetition=repetition,physical_image_id=r['key'],independent_test_eligible=False)
    for split,count in cfg['negative_counts'].items():
        for index in range(count):
            key=f'negative_{cfg["seed"]}_{split}_{index:04d}'
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
            image,kind=negative_image(seed)
            image_path=out/'images'/split/(key+'.jpg');label_path=out/'labels'/split/(key+'.txt')
            image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
            if not image_path.exists():image.save(image_path,quality=95,subsampling=0)
            if label_path.exists() and label_path.read_text(encoding='utf-8').strip():raise ValueError('Negative label must be empty')
            label_path.touch(exist_ok=True)
            rows.append({'key':key,'domain':'negative','split':split,'group_id':key,'image':image_path.relative_to(out).as_posix(),
                         'label':label_path.relative_to(out).as_posix(),'image_sha256':sha(image_path),'label_sha256':sha(label_path),
                         'classes':[],'mode':kind,'seed':seed,'synthetic':True,'no_sorting_objects_rendered':True})
    manifest=out/'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    counts=Counter(f"{r['domain']}_{r['split']}" for r in rows)
    frozen={**inputs,'manifest_sha256':sha(manifest),'counts':dict(counts),'unique_real_training_photos':46,
            'real_training_repetitions':cfg['real_repetitions'],'real_independent_test_images':0,
            'note':'Repeated real photos remain same training group; never count copies as new real samples. No prior test images included.'}
    (out/'freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'data.yaml').write_text(yaml.safe_dump({'path':out.resolve().as_posix(),'train':'images/train','val':'images/val','names':NAMES,'nc':9},allow_unicode=True),encoding='utf-8')
    return verify(out)


def verify(out):
    frozen=json.loads((out/'freeze.json').read_text(encoding='utf-8'))
    if frozen['config_sha256']!=sha(CONFIG) or frozen['manifest_sha256']!=sha(out/'manifest.jsonl'):raise ValueError('Frozen dataset changed')
    seen_hash={};seen_group={};counts=Counter()
    for line in (out/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);counts[f"{r['domain']}_{r['split']}"]+=1
        for kind in ['image','label']:
            if sha(out/r[kind])!=r[kind+'_sha256']:raise ValueError('Changed '+r[kind])
        for key,seen in [(r['image_sha256'],seen_hash),(r['group_id'],seen_group)]:
            if key in seen and seen[key]!=r['split']:raise ValueError('Cross-split leakage')
            seen[key]=r['split']
        if r['domain']=='real' and r['split']!='train':raise ValueError('Real group leaked into validation')
    if dict(counts)!=frozen['counts']:raise ValueError('Count mismatch')
    return frozen


if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False,indent=2))
