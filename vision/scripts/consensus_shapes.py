"""Class-blind agreement between two detectors; no dataset or truth access."""
from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_ious(primary, secondary):
    result = np.zeros((len(primary), len(secondary)), dtype=float)
    for i, a in enumerate(primary):
        ax0, ay0, ax1, ay1 = a['box']
        aa = max(0., ax1-ax0)*max(0., ay1-ay0)
        for j, b in enumerate(secondary):
            bx0, by0, bx1, by1 = b['box']
            ba = max(0., bx1-bx0)*max(0., by1-by0)
            overlap = max(0., min(ax1,bx1)-max(ax0,bx0))*max(0., min(ay1,by1)-max(ay0,by0))
            result[i,j] = overlap/max(1e-12, aa+ba-overlap)
    return result


def select_predictions(primary, secondary, policy, overlaps=None):
    """Return retained predictions and explicit (source,index) mask references.

    Agreement is geometric, independent of shape labels. Matched detections use
    primary geometry. Unmatched detections require a fixed confidence threshold;
    colliding secondary detections cannot reintroduce the same object.
    """
    threshold = float(policy['match_iou'])
    if not 0 < threshold <= 1:
        raise ValueError('match_iou must be within (0,1]')
    mode = policy['class_source']
    if mode not in {'primary','stronger_secondary'}:
        raise ValueError('Unsupported class source')
    overlap = box_ious(primary, secondary) if overlaps is None else np.asarray(overlaps)
    if overlap.shape != (len(primary),len(secondary)):
        raise ValueError('Prediction/overlap alignment changed')
    matches = {}
    if overlap.size:
        valid = overlap >= threshold
        ii,jj = linear_sum_assignment(-(valid*(min(overlap.shape)+1)+overlap*valid))
        matches = {int(i):int(j) for i,j in zip(ii,jj) if valid[i,j]}
    selected = []
    for i,prediction in enumerate(primary):
        j = matches.get(i)
        if j is None and prediction['confidence'] < policy['unmatched_primary_confidence']:
            continue
        revised = copy.deepcopy(prediction)
        if j is not None and mode == 'stronger_secondary':
            if secondary[j]['confidence'] > prediction['confidence'] + policy.get('secondary_margin',.1):
                revised['class_id'] = secondary[j]['class_id']
        revised['consensus_source'] = 'primary'
        revised['consensus_source_index'] = i
        revised['consensus_partner_index'] = j
        selected.append(revised)
    used_secondary = set(matches.values())
    for j,prediction in enumerate(secondary):
        if j in used_secondary or prediction['confidence'] < policy['unmatched_secondary_confidence']:
            continue
        if len(primary) and overlap[:,j].max() >= threshold:
            continue
        revised = copy.deepcopy(prediction)
        revised['consensus_source'] = 'secondary'
        revised['consensus_source_index'] = j
        revised['consensus_partner_index'] = None
        selected.append(revised)
    return selected
