"""Audit source preservation, binary draft masks and unchanged deployment artifacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from import_real_review import contained, validate_review

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    out=ROOT/'dataset/real/annotations/review_v2'
    data=json.loads((out/'annotations.json').read_text(encoding='utf-8'))
    ids=set();hashes=set();classes=Counter();corrected=0;instances=0
    archives=[]
    for name in ['batch_20261101_import','batch_20261102_capture']:
        archives+=json.loads((ROOT/'dataset/real/raw'/name/'manifest.json').read_text(encoding='utf-8'))['images']
    for row in archives:
        source=ROOT/row['source'];dest=ROOT/row['destination']
        assert source.read_bytes()==dest.read_bytes()
        assert sha(dest)==row['sha256']
    assert len(archives)==47 and len({r['sha256'] for r in archives})==46
    for r in data['images']:
        assert r['image_id'] not in ids and r['source_sha256'] not in hashes
        ids.add(r['image_id']);hashes.add(r['source_sha256'])
        assert json.loads((out/'records'/f'{r["image_id"]}.json').read_text(encoding='utf-8'))==r
        assert sha(ROOT/r['source'])==r['source_sha256']
        assert not r['training_approved'] and not r['final_test_eligible'] and r['human_review_status']=='pending'
        assert r['split']=='development_unassigned' and r['leakage_group']=='real_pilot_shared_physical_objects'
        with Image.open(out/r['image']) as im:assert list(im.size)==r['annotation_size']
        assert (len(r['instances'])==0)==r['is_empty_tray']
        for i in r['instances']:
            instances+=1;classes[i['class_name']]+=1;corrected+=bool(i.get('assistant_correction'))
            assert type(i['class_id']) is int and 0<=i['class_id']<9
            assert not i['boundary_human_approved'] and not i['warnings']
            assert sha(out/i['mask'])==i['mask_sha256']
            with Image.open(out/i['mask']) as im:
                assert list(im.size)==r['annotation_size'];pixels=np.asarray(im)
            assert set(np.unique(pixels))<={0,255} and np.count_nonzero(pixels)>20
            assert np.count_nonzero(pixels)==i['mask_foreground_pixels']
    expected={
        'models/trained/stl_shape_v1/best.pt':'5c7dd29d84c0b71b3110ca0da0f101210648e85b7e9badded00c2ab09f3b057e',
        'models/trained/stl_shape_focus_v2/best.pt':'d51cc9c95e9949691f9aafc8bf7efd5adf746b2578f3e7a0b755553a127f36e0',
        'models/trained/stl_shape_focus_v2/pipeline.json':'eb849210fcbeea2437f6ebc917aeb148af334fe7794fba1ac5ee0c0b1159b410'}
    for path,value in expected.items():assert sha(ROOT/path)==value
    approved=0;provenance=None
    if (out/'current_review.json').exists():
        pointer=json.loads((out/'current_review.json').read_text(encoding='utf-8'))
        snapshot=contained(out,pointer['snapshot']);assert sha(snapshot)==pointer['snapshot_sha256']
        accepted=json.loads(snapshot.read_text(encoding='utf-8'))
        reviewed=validate_review(data,accepted['export'])
        assert reviewed['reviewed_image_count']==pointer['approved_images']
        approved=reviewed['reviewed_image_count'];provenance=accepted['export'].get('approval_provenance')
    result={'source_files':47,'unique_images':len(ids),'duplicate_excluded':1,'empty_images':sum(r['is_empty_tray'] for r in data['images']),
        'instances':instances,'class_counts':dict(classes),'assistant_corrected_masks':corrected,'source_bytes_preserved':True,
        'approved_images':approved,'approval_provenance':provenance,'all_training_approved_false':True,'final_test_eligible':False,'deployment_artifacts_unchanged':True,
        'review_revision':data['review_revision'],'scope':'Structural audit plus recorded explicit user approvals; not an independent model accuracy evaluation.'}
    audit_path=ROOT/'runs/task_audit/2026-11-01_real_review_data.json'
    audit_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
