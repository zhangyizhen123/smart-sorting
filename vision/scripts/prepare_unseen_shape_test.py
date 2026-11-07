"""Prepare 2900 new synthetic test scenes; never run a model or inspect predictions."""
from collections import Counter
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from make_focused_dataset import render_scene,CLASSES,MODEL_DIR,load_binary_stl,make_sphere_proxy
from prepare_replay_optimization import negative_image

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'vision/configs/unseen_shape_test_v3.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8');temp.replace(path)


def verify(root):
    frozen=json.loads((root/'freeze.json').read_text(encoding='utf-8'))
    if frozen['config_sha256']!=sha(CONFIG) or frozen['manifest_sha256']!=sha(root/'manifest.jsonl'):raise ValueError('Sealed test changed')
    counts=Counter();keys=set();images=set()
    for line in (root/'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['split']!='test' or r['key'] in keys:raise ValueError('Invalid held-out group')
        keys.add(r['key']);counts[r['domain']]+=1
        for kind in ['image','label']:
            if sha(root/r[kind])!=r[kind+'_sha256']:raise ValueError('Changed test asset')
        if r['image_sha256'] in images:raise ValueError('Duplicate test image')
        images.add(r['image_sha256'])
    if counts!=Counter(legacy=900,focused=1800,negative=200):raise ValueError('Unexpected sealed test size')
    # No new test image may be an exact copy of any available development image.
    other_manifests=[ROOT/'dataset/replay_optimization_v3/manifest.jsonl',ROOT/'dataset/focused_v2/manifest.jsonl']
    for manifest in other_manifests:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if json.loads(line)['image_sha256'] in images:raise ValueError('Test/development hash overlap')
    return frozen


def prepare():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output']
    if (out/'freeze.json').exists():return verify(out)
    out.mkdir(parents=True,exist_ok=True)
    lock={'config_sha256':sha(CONFIG),'sources':{name:sha(ROOT/'vision/scripts'/name) for name in
         ['prepare_unseen_shape_test.py','make_focused_dataset.py','make_stl_synthetic.py','prepare_replay_optimization.py']}}
    source_lock=out/'build_inputs.json'
    if source_lock.exists() and json.loads(source_lock.read_text(encoding='utf-8'))!=lock:raise ValueError('Test generation code changed')
    write_json(source_lock,lock);rows=[]
    legacy=out/'legacy'
    if not (legacy/'manifest.csv').exists():
        subprocess.run([sys.executable,str(ROOT/'vision/scripts/make_stl_synthetic.py'),'--out',str(legacy),
                        '--per-class',str(cfg['legacy_per_class']),'--image-size','320','--seed',str(cfg['seed'])],check=True)
    with (legacy/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            image=legacy/'images'/r['image'];label=legacy/'labels'/Path(r['image']).with_suffix('.txt').name;key=f'legacy_unseen_{cfg["seed"]}_'+Path(r['image']).stem
            classes=[int(line.split()[0]) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
            rows.append({'key':key,'domain':'legacy','split':'test','group_id':key,'image':image.relative_to(out).as_posix(),
                         'label':label.relative_to(out).as_posix(),'image_sha256':sha(image),'label_sha256':sha(label),
                         'classes':classes,'seed':cfg['seed'],'target_class':r['target_class']})
    meshes={name:load_binary_stl(MODEL_DIR/f'{name}.stl') for name in CLASSES};meshes['sphere']=make_sphere_proxy(meshes['sphere'])
    for target in range(9):
        for index in range(cfg['focused_per_class']):
            key=f'focused_unseen_v3_{cfg["seed"]}_{target:02d}_{index:05d}'
            record_path=out/'generation_records'/(key+'.json')
            if record_path.exists():
                record=json.loads(record_path.read_text(encoding='utf-8'))
                for kind in ['image','label']:
                    if sha(out/record[kind])!=record[kind+'_sha256']:raise ValueError('Partial test asset changed')
            else:
                seed=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
                image,instances,meta=render_scene(meshes,target,index,640,seed)
                image_path=out/'focused/images'/f'{key}.jpg';label_path=out/'focused/labels'/f'{key}.txt'
                image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
                image.save(image_path,quality=95,subsampling=0)
                lines=[str(cls)+' '+' '.join(f'{v:.7f}' for v in (poly.astype(float)/640).ravel()) for cls,poly in instances]
                label_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
                record={'key':key,'domain':'focused','split':'test','group_id':key,'image':image_path.relative_to(out).as_posix(),
                        'label':label_path.relative_to(out).as_posix(),'image_sha256':sha(image_path),'label_sha256':sha(label_path),
                        'classes':[cls for cls,_ in instances],'target_class':CLASSES[target],**meta}
                write_json(record_path,record)
            rows.append(record)
        print(json.dumps({'stage':'focused_test_generation','class':CLASSES[target],'images':cfg['focused_per_class']}),flush=True)
    for index in range(cfg['negative_images']):
        key=f'negative_unseen_{cfg["seed"]}_{index:04d}';seed=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
        image_path=out/'negative/images'/f'{key}.jpg';label_path=out/'negative/labels'/f'{key}.txt'
        image_path.parent.mkdir(parents=True,exist_ok=True);label_path.parent.mkdir(parents=True,exist_ok=True)
        image,mode=negative_image(seed)
        if not image_path.exists():image.save(image_path,quality=95,subsampling=0)
        if label_path.exists() and label_path.read_text(encoding='utf-8').strip():raise ValueError('Negative label is not empty')
        label_path.touch(exist_ok=True)
        rows.append({'key':key,'domain':'negative','split':'test','group_id':key,'image':image_path.relative_to(out).as_posix(),
                     'label':label_path.relative_to(out).as_posix(),'image_sha256':sha(image_path),'label_sha256':sha(label_path),
                     'classes':[],'seed':seed,'mode':mode,'no_sorting_objects_rendered':True})
    manifest=out/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    frozen={**lock,'manifest_sha256':sha(manifest),'counts':dict(Counter(r['domain'] for r in rows)),'images':len(rows),
            'independent_real_images':0,'candidate_locked_before_evaluation_required':True,'predictions_generated':False}
    write_json(out/'freeze.json',frozen);return verify(out)


if __name__=='__main__':print(json.dumps(prepare(),ensure_ascii=False,indent=2))
