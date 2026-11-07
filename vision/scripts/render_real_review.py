"""Render the local human-review interface without mutating source photographs."""
from pathlib import Path
import json
from import_real_review import contained, digest, validate_review

ROOT=Path(__file__).resolve().parents[2]


def render(payload,output):
    payload=dict(payload)
    current=output/'current_review.json'
    if current.exists():
        pointer=json.loads(current.read_text(encoding='utf-8'))
        path=contained(output,pointer['snapshot'])
        if pointer['review_revision']!=payload['review_revision'] or digest(path)!=pointer['snapshot_sha256']:
            raise ValueError('Current review snapshot is stale or changed')
        snapshot=json.loads(path.read_text(encoding='utf-8'))
        reviewed=validate_review(payload,snapshot['export'])
        payload['saved_review']={'snapshot_sha256':pointer['snapshot_sha256'],
            'reviewed_image_count':reviewed['reviewed_image_count'],
            'reviews':snapshot['export']['reviews'],
            'approval_provenance':snapshot['export'].get('approval_provenance')}
    template=(ROOT/'vision/templates/real_review.html').read_text(encoding='utf-8')
    embedded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')
    (output/'index.html').write_text(template.replace('__ANNOTATIONS__',embedded),encoding='utf-8',newline='\n')


if __name__=='__main__':
    out=ROOT/'dataset/real/annotations/review_v2'
    render(json.loads((out/'annotations.json').read_text(encoding='utf-8')),out)
