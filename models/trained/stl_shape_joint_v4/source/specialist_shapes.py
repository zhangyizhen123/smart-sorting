"""Class-only consultation of a confidence-qualified specialist detector."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from consensus_shapes import box_ious


def specialist_matches(primary,secondary,threshold=.5):
    overlap=box_ious(primary,secondary)
    if not overlap.size:
        return {}
    valid=overlap>=threshold
    confidence=np.asarray([p['confidence'] for p in secondary])
    score=valid*(min(overlap.shape)+1)+overlap*confidence[None,:]*valid
    ii,jj=linear_sum_assignment(-score)
    return {int(i):int(j) for i,j in zip(ii,jj) if valid[i,j]}


def consult(primary,secondary,policy,matches=None):
    matches=specialist_matches(primary,secondary,policy['match_iou']) if matches is None else matches
    result=copy.deepcopy(primary)
    for i,j in matches.items():
        a,b=primary[i],secondary[j]
        if b['class_id'] not in policy['allowed_classes']:
            continue
        if (a['confidence']<policy['maximum_primary_confidence'] and
            b['confidence']>=policy['minimum_specialist_confidence'] and
            b['confidence']>=a['confidence']+policy['confidence_margin']):
            result[i]['class_id']=b['class_id']
            result[i]['specialist_source_index']=j
    return result
