"""Validate an explicit human review export and retain an immutable review snapshot.

Does not start training, assign splits, or convert draft masks to YOLO polygons.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'dataset/real/annotations/review_v2'
NAMES = ['cylinder','cube','cuboid','tri_prism','pent_prism','hex_prism','cone','tetrahedron','sphere']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Path escapes annotation/source directory')
    return path


def validate_review(data, exported):
    if exported.get('version') != 1 or exported.get('source') != 'explicit_user_review':
        raise ValueError('Unsupported review export')
    if exported.get('review_revision') != data['review_revision']:
        raise ValueError('Review revision is stale; reopen the current review page')
    records = exported.get('reviews')
    if not isinstance(records, list):
        raise ValueError('Missing review list')
    ids = [r.get('image_id') for r in records]
    if len(ids) != len(set(ids)) or set(ids) != {r['image_id'] for r in data['images']}:
        raise ValueError('Missing, repeated, or unknown image IDs')
    by_id = {r['image_id']:r for r in records}
    reviewed = copy.deepcopy(data)
    approved = 0
    for image in reviewed['images']:
        r = by_id[image['image_id']]
        if r.get('image_sha256') != image['source_sha256']:
            raise ValueError('Image hash mismatch')
        if type(r.get('approved')) is not bool:
            raise ValueError('Approval must be an explicit boolean')
        count = r.get('actual_count')
        if type(count) is not int or not 0 <= count <= 100 or not isinstance(r.get('notes'),str):
            raise ValueError('Invalid count or notes')
        instances = r.get('instances')
        if not isinstance(instances,list) or len(instances) != len(image['instances']):
            raise ValueError('Instance list is incomplete')
        if r['approved'] and count != len(instances):
            raise ValueError('Cannot approve an incomplete instance count')
        for expected, actual in zip(image['instances'],instances,strict=True):
            if actual.get('instance_id') != expected['instance_id'] or actual.get('mask_sha256') != expected['mask_sha256']:
                raise ValueError('Instance ID or mask hash mismatch')
            cls = actual.get('class_name')
            if cls not in NAMES and not (cls == 'unknown' and not r['approved']):
                raise ValueError('Approved instances require known classes')
            expected['review_proposed_class_name'] = cls
            expected['boundary_human_approved'] = r['approved']
            if r['approved']:
                expected.update(class_name=cls,class_id=NAMES.index(cls),class_status='human_reviewed')
        image.update(human_review_status='approved' if r['approved'] else 'pending',
                     review_notes=r['notes'],review_actual_count=count,
                     training_approved=False,final_test_eligible=False)
        approved += r['approved']
    reviewed['reviewed_image_count'] = approved
    reviewed['scope'] = 'Human-reviewed references where explicitly approved; split and training preparation remain separate.'
    return reviewed


def import_review(path, annotation_dir=DEFAULT, *, check_only=False):
    data = json.loads((annotation_dir/'annotations.json').read_text(encoding='utf-8'))
    exported = json.loads(path.read_text(encoding='utf-8-sig'))
    reviewed = validate_review(data,exported)
    for r in data['images']:
        if digest(contained(ROOT,r['source'])) != r['source_sha256']:
            raise ValueError('Source image changed since annotation')
        for instance in r['instances']:
            if digest(contained(annotation_dir,instance['mask'])) != instance['mask_sha256']:
                raise ValueError('Mask file changed since review page generation')
    export_hash = digest(path)
    reviewed['review_export_sha256'] = export_hash
    target = annotation_dir/'accepted_reviews'/f'{export_hash}.json'
    payload = json.dumps({'export':exported,'reviewed_annotations':reviewed},ensure_ascii=False,indent=2)+'\n'
    if not check_only:
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if target.read_text(encoding='utf-8') != payload:
                raise FileExistsError('Conflicting review snapshot')
        else:
            with target.open('x',encoding='utf-8') as f: f.write(payload)
        # Point the review UI to this validated immutable snapshot, without editing drafts.
        current={'version':1,'review_revision':data['review_revision'],
                 'snapshot':target.relative_to(annotation_dir).as_posix(),
                 'snapshot_sha256':digest(target),'approved_images':reviewed['reviewed_image_count']}
        temporary=annotation_dir/'current_review.json.tmp'
        temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temporary.replace(annotation_dir/'current_review.json')
    return {'images':len(data['images']),'explicitly_approved':reviewed['reviewed_image_count'],
            'training_started':False,'check_only':check_only,'snapshot':str(target) if not check_only else None}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('review',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();print(json.dumps(import_review(a.review,check_only=a.check_only),ensure_ascii=False,indent=2))
