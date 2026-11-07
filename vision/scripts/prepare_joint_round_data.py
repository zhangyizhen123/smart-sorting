"""Generate versioned development or held-out scenes without running inference."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from make_focused_dataset import render_scene, CLASSES, MODEL_DIR, load_binary_stl, make_sphere_proxy
from prepare_replay_optimization import negative_image

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify(config_path):
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    out = ROOT/cfg['output']; freeze = json.loads((out/'freeze.json').read_text())
    if digest(config_path) != freeze['config_sha256'] or digest(out/'manifest.jsonl') != freeze['manifest_sha256']:
        raise ValueError('Frozen dataset/config changed')
    rows = [json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
    images = set(); keys = set()
    for row in rows:
        if row['split'] != cfg['split'] or row['key'] in keys:
            raise ValueError('Invalid split or duplicate key')
        keys.add(row['key'])
        for kind in ['image','label']:
            if digest(out/row[kind]) != row[kind+'_sha256']:
                raise ValueError('Dataset asset changed')
        if row['image_sha256'] in images:
            raise ValueError('Duplicate image within this dataset')
        images.add(row['image_sha256'])
    expected = Counter(legacy=9*cfg['legacy_per_class'], focused=9*cfg['focused_per_class'], negative=cfg['negative_images'])
    if Counter(row['domain'] for row in rows) != expected:
        raise ValueError('Incomplete dataset')
    for relative in cfg['exclude_manifests']:
        if digest(ROOT/relative) != freeze['excluded_manifest_sha256'][relative]:
            raise ValueError('Excluded data identity changed')
        for line in (ROOT/relative).read_text(encoding='utf-8').splitlines():
            if json.loads(line)['image_sha256'] in images:
                raise ValueError('Cross-dataset duplicate image')
    return freeze


def prepare(config_path):
    cfg = json.loads(config_path.read_text(encoding='utf-8')); out = ROOT/cfg['output']
    if cfg['role'] not in {'development','sealed_test'} or cfg['split'] != ('val' if cfg['role']=='development' else 'test'):
        raise ValueError('Dataset role/split mismatch')
    if (out/'freeze.json').exists():
        return verify(config_path)
    inputs = {'config_sha256':digest(config_path), 'generator_sha256':digest(Path(__file__)),
              'source_sha256':{n:digest(ROOT/'vision/scripts'/n) for n in ['make_stl_synthetic.py','make_focused_dataset.py','prepare_replay_optimization.py']},
              'mesh_sha256':{name:digest(MODEL_DIR/(name+'.stl')) for name in CLASSES}}
    if (out/'build_inputs.json').exists() and json.loads((out/'build_inputs.json').read_text()) != inputs:
        raise ValueError('Partial generation has different inputs')
    save(out/'build_inputs.json',inputs); rows=[]; legacy=out/'legacy'
    if not (legacy/'manifest.csv').exists():
        subprocess.run([sys.executable,str(ROOT/'vision/scripts/make_stl_synthetic.py'),'--out',str(legacy),
                        '--per-class',str(cfg['legacy_per_class']),'--image-size','320','--seed',str(cfg['seed'])],check=True)
    with (legacy/'manifest.csv').open(encoding='utf-8-sig',newline='') as handle:
        for source in csv.DictReader(handle):
            image=legacy/'images'/source['image']; label=legacy/'labels'/Path(source['image']).with_suffix('.txt').name
            key=f"legacy_round4_{cfg['seed']}_{image.stem}"
            rows.append(dict(key=key,domain='legacy',split=cfg['split'],group_id=key,image=image.relative_to(out).as_posix(),
                             label=label.relative_to(out).as_posix(),image_sha256=digest(image),label_sha256=digest(label),
                             classes=[int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()],seed=cfg['seed']))
    meshes={name:load_binary_stl(MODEL_DIR/(name+'.stl')) for name in CLASSES}
    meshes['sphere']=make_sphere_proxy(meshes['sphere'])
    for target in range(9):
        for index in range(cfg['focused_per_class']):
            key=f"focused_round4_{cfg['seed']}_{target:02d}_{index:05d}"
            record_path=out/'generation_records'/(key+'.json')
            if record_path.exists():
                row=json.loads(record_path.read_text())
                if any(digest(out/row[k])!=row[k+'_sha256'] for k in ['image','label']):
                    raise ValueError('Partial generated asset changed')
            else:
                seed=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
                image,instances,meta=render_scene(meshes,target,index,640,seed)
                ip=out/'focused/images'/(key+'.jpg');lp=out/'focused/labels'/(key+'.txt')
                ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
                image.save(ip,quality=95,subsampling=0)
                lp.write_text('\n'.join(str(cls)+' '+' '.join(f'{v:.7f}' for v in (poly.astype(float)/640).ravel()) for cls,poly in instances)+'\n')
                row=dict(key=key,domain='focused',split=cfg['split'],group_id=key,image=ip.relative_to(out).as_posix(),
                         label=lp.relative_to(out).as_posix(),image_sha256=digest(ip),label_sha256=digest(lp),
                         classes=[cls for cls,_ in instances],target_class=CLASSES[target],**meta)
                save(record_path,row)
            rows.append(row)
        print(json.dumps(dict(stage='generate_development' if cfg['role']=='development' else 'generate_sealed_test',shape=CLASSES[target],images=len(rows))),flush=True)
    for index in range(cfg['negative_images']):
        key=f"negative_round4_{cfg['seed']}_{index:04d}";seed=int(hashlib.sha256(key.encode()).hexdigest()[:12],16)
        ip=out/'negative/images'/(key+'.jpg');lp=out/'negative/labels'/(key+'.txt')
        ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        image,mode=negative_image(seed)
        if not ip.exists(): image.save(ip,quality=95,subsampling=0)
        if lp.exists() and lp.read_text().strip(): raise ValueError('Negative labels must be empty')
        lp.touch(exist_ok=True)
        rows.append(dict(key=key,domain='negative',split=cfg['split'],group_id=key,image=ip.relative_to(out).as_posix(),
                         label=lp.relative_to(out).as_posix(),image_sha256=digest(ip),label_sha256=digest(lp),classes=[],seed=seed,mode=mode))
    manifest=out/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',encoding='utf-8')
    frozen=dict(**inputs,manifest_sha256=digest(manifest),images=len(rows),role=cfg['role'],independent_real_images=0,
                counts=dict(Counter(row['domain'] for row in rows)),excluded_manifest_sha256={p:digest(ROOT/p) for p in cfg['exclude_manifests']})
    save(out/'freeze.json',frozen)
    return verify(config_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);args=parser.parse_args()
    print(json.dumps(prepare(ROOT/args.config),ensure_ascii=False),flush=True)
