"""Strict research scoring shared by the JEV-L2/L3 offline reports.

No answer/delegate compatibility: frozen historical scorer retains that rule.
The version-specific CLI scripts own truth review and report shapes.
"""
from __future__ import annotations
from collections import Counter


def label_stats(truth, predictions, labels):
    """Return strict set-valued accuracy and per-label P/R/F1.

    Predictions for a multi-label answer+skip case count as correct for either
    label. Both labels remain in ground-truth support, independent of system.
    """
    if set(truth)!=set(predictions):
        raise ValueError(f'ID mismatch: {len(set(truth)-set(predictions))} missing, {len(set(predictions)-set(truth))} extra')
    n=len(truth)
    correct=sum(predictions[cid] in ls for cid,ls in truth.items())
    per={}
    for label in labels:
        tp=sum(predictions[cid]==label and label in ls for cid,ls in truth.items())
        fp=sum(predictions[cid]==label and label not in ls for cid,ls in truth.items())
        fn=sum(label in ls and predictions[cid]!=label for cid,ls in truth.items())
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else None
        f1=2*precision*recall/(precision+recall) if recall and precision+recall else 0
        per[label]={'support':tp+fn,'precision':precision,'recall':recall,'f1':f1}
    present=[v['f1'] for v in per.values() if v['support']]
    return {'n':n,'correct':correct,'accuracy':correct/n if n else None,
            'macro_f1':sum(present)/len(present) if present else None,
            'truth_distribution':dict(Counter('+'.join(ls) for ls in truth.values())),
            'per_label':per,
            'mistakes':[{'case_id':cid,'truth':ls,'prediction':predictions[cid]}
                        for cid,ls in truth.items() if predictions[cid] not in ls]}
