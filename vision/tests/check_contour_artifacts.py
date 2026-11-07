"""Verify all frozen audit records and mask hashes, plus previous 640 baselines."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vision/scripts'))
from contour_quality import mask_iou, topology


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg_path=ROOT/'vision/configs/contour_audit_v1.json'
    cfg=json.loads(cfg_path.read_text(encoding='utf-8'));out=ROOT/cfg['output']
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    lock=json.loads((out/'freeze.json').read_text(encoding='utf-8'))
    assert sha(cfg_path)==lock['config_sha256']
    for name,path in cfg['models'].items():assert sha(ROOT/path)==lock['models'][name]
    for name,digest in lock['sources'].items():assert sha(ROOT/'vision/scripts'/name)==digest
    assert summary['image_runs']==976 and summary['independent_real_test_images']==0
    assert summary['no_training'] and not summary['automatic_deployment']
    records_count=files_count=accepted=guarded=0;domain_counts={}
    for variant in summary['variants']:
        paths=sorted((out/'records'/variant).glob('*.json'));assert len(paths)==244
        domains=Counter()
        # First/middle/last plus a known dense scene cover coordinate and topology checks.
        representatives={0,len(paths)//2,len(paths)-1}
        for index,path in enumerate(paths):
            r=json.loads(path.read_text(encoding='utf-8'));records_count+=1;domains[r['domain']]+=1
            n=len(r['raw']['predictions']);assert n==len(r['geometry'])==len(r['regularized']['predictions'])
            assert r['raw']['predictions']==r['regularized']['predictions']
            assert len(r['mask_hashes'])==n*2
            for asset,digest in r['mask_hashes'].items():
                assert sha(out/asset)==digest,asset;files_count+=1
            for item in r['geometry']:
                assert item['effective_iou_to_raw']>=.98
                if item['geometry_status']=='accepted':
                    accepted+=1
                    assert item['candidate_iou']>=.98
                    assert item['candidate_area_change']<=.02
                    assert item['candidate_center_shift_pixels']<=item['epsilon_pixels']
                else:
                    guarded+=1;assert item['changed_pixels']==0
            if index in representatives or r['confirmed_occlusion_example']:
                raws=[];regs=[]
                for j,meta in enumerate(r['geometry']):
                    directory=out/'masks'/variant/r['key']
                    raw=np.asarray(Image.open(directory/'raw'/f'{j+1:02d}.png'))>0
                    reg=np.asarray(Image.open(directory/'regularized'/f'{j+1:02d}.png'))>0
                    assert raw.shape==reg.shape==(r['height'],r['width'])
                    assert abs(mask_iou(raw,reg)-meta['effective_iou_to_raw'])<1e-12
                    assert topology(raw.astype(np.uint8))==topology(reg.astype(np.uint8))
                    raws.append(raw);regs.append(reg)
                for i in range(n):
                    for j in range(i+1,n):assert not np.any((regs[i]&regs[j])&~(raws[i]&raws[j]))
        assert domains==Counter(real=46,legacy=108,focused=90)
        domain_counts[variant]=dict(domains)
    previous=ROOT/'runs/segment/real_mixed_pilot_v1/evaluation'
    for variant,origin in [('initial_640','from_v1_initial'),('pilot_640','from_v1_best')]:
        real=json.loads((previous/f'real_train_fit_{origin}.json').read_text(encoding='utf-8'))
        val=json.loads((previous/f'val_{origin}.json').read_text(encoding='utf-8'))
        assert abs(summary['variants'][variant]['real']['raw']['0.5']['strict_instance_score']-real['overall']['strict_instance_score'])<1e-12
        for domain in ['legacy','focused']:
            assert abs(summary['variants'][variant][domain]['raw']['0.5']['strict_instance_score']-val['by_domain'][domain]['strict_instance_score'])<1e-12
    result={'records':records_count,'mask_files_hashed':files_count,'accepted_geometry':accepted,
            'guarded_fallbacks':guarded,'domain_counts':domain_counts,'previous_640_baselines_exact':True,
            'model_and_source_hashes_unchanged':True,'representative_mask_topology_and_overlap_checked':True}
    target=ROOT/'runs/task_audit/2026-11-01_contour_audit_artifacts.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
