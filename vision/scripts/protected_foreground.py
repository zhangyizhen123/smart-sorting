"""Retain strong detector-supported objects when a background gate is uncertain.

Inputs contain only predictions/probabilities; no labels, filenames or domains.
"""
import math


def retained_indices(predictions, probabilities, policy):
    if len(predictions) != len(probabilities):
        raise ValueError('Prediction/foreground probability alignment mismatch')
    threshold=float(policy['foreground_threshold']); confidence=float(policy['protect_confidence'])
    if not 0<threshold<1 or not 0<=confidence<=1.01:
        raise ValueError('Invalid foreground protection policy')
    selected=[]; protected=[]
    for index,(prediction,probability) in enumerate(zip(predictions,probabilities,strict=True)):
        if not math.isfinite(float(probability)) or not 0<=probability<=1:
            raise ValueError('Invalid foreground probability')
        supported=(prediction.get('consensus_source')=='primary'
                   and prediction['confidence']>=confidence
                   and prediction.get('verifier_background_probability',1.)<=.5
                   and (not policy['require_partner'] or prediction.get('consensus_partner_index') is not None))
        if probability>=threshold or supported:
            selected.append(index);protected.append(bool(probability<threshold and supported))
    return selected,protected
