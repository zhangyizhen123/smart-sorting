"""Preserve each detector's evidence for development and reproducible arbitration."""
import copy
from consensus_shapes import select_predictions


def add_evidence(item,primary,secondary):
    result=copy.deepcopy(item)
    source=primary if item['consensus_source']=='primary' else secondary
    result['pre_consensus_class_id']=source[item['consensus_source_index']]['class_id']
    partner=item.get('consensus_partner_index')
    if item['consensus_source']=='primary' and partner is not None:
        result['partner_class_id']=secondary[partner]['class_id']
        result['partner_confidence']=secondary[partner]['confidence']
    else:
        result['partner_class_id']=None;result['partner_confidence']=None
    return result


def combine_native(results,policy):
    a,b=results['primary'],results['secondary']
    selection=select_predictions(a[0],b[0],policy)
    predictions=[];masks=[];originals=[]
    for selected in selection:
        item=add_evidence(selected,a[0],b[0])
        source=results[item['consensus_source']];index=item['consensus_source_index']
        original_index=source[0][index]['source_prediction_index']
        originals.append(source[2][original_index]);masks.append(source[1][index])
        item['native_source_prediction_index']=source[0][index].get('native_source_prediction_index',original_index)
        item['source_prediction_index']=len(predictions);predictions.append(item)
    return predictions,masks,originals,a[3]


def reclassify(predictions,margin):
    if not 0<=margin<=1.01: raise ValueError('Invalid class-agreement margin')
    result=copy.deepcopy(predictions)
    for item in result:
        item['class_id']=item['pre_consensus_class_id']
        if item['partner_confidence'] is not None and item['partner_confidence']>item['confidence']+margin:
            item['class_id']=item['partner_class_id']
    return result
