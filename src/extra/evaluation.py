import torch

# pred : prediction
# gt : ground truth

def get_tf_pn(pred, gt, ts=[]):
    ts = [t.lower() for t in ts]
    res = []
    
    # true-positive
    if 'tp' in ts:
        tp = ((pred==1)+(gt==1))==2
        res.append(tp)
    
    # true-negetive
    if 'tn' in ts:
        tn = ((pred==0)+(gt==0))==2
        res.append(tn)
    
    # false-positive
    if 'fp' in ts:
        fp = ((pred==1)+(gt==0))==2
        res.append(fp)
    
    # false-negetive
    if 'fn' in ts:
        fn = ((pred==0)+(gt==1))==2
        res.append(fn)
         

def get_accuracy(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    corr = torch.sum(pred==gt)
    tensor_size = pred.size(0)*pred.size(1)*pred.size(2)*pred.size(3)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(pred, gt, threshold=0.5):
    # Sensitivity == Recall
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tp, fn = get_tf_pn(pred, gt, ['tp', 'fn'])
    se = float(torch.sum(tp))/(float(torch.sum(tp+fn)) + 1e-6)     
    return se


def get_specificity(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tn, fp = get_tf_pn(pred, gt, ['tn', 'fp'])
    sp = float(torch.sum(tn))/(float(torch.sum(tn+fp)) + 1e-6)
    return sp


def get_precision(pred, gt, threshold=0.5):
    pred = pred > threshold
    gt = gt == torch.max(gt)
    tp, fp = get_tf_pn(pred, gt, ['tp', 'fp'])
    pc = float(torch.sum(tp))/(float(torch.sum(tp+fp)) + 1e-6)
    return pc


def get_f1(pred, gt, threshold=0.5):
    # Sensitivity == Recall
    se = get_sensitivity(pred, gt, threshold=threshold)
    pc = get_precision(pred, gt, threshold=threshold)
    f1 = 2*se * pc/(se+pc + 1e-6)
    return f1


def get_js(pred, gt, threshold=0.5):
    # js : jaccard similarity
    pred = pred > threshold
    gt = gt == torch.max(gt)
    Inter = torch.sum((pred+gt)==2)
    Union = torch.sum((pred+gt)>=1)
    js = float(Inter)/(float(Union) + 1e-6)
    return js


def get_dc(pred,gt,threshold=0.5):
    # dc : dice coefficient
    pred = pred > threshold
    gt = gt == torch.max(gt)
    Inter = torch.sum((pred+gt)==2)
    dc = float(2*Inter)/(float(torch.sum(pred)+torch.sum(gt)) + 1e-6)
    return dc