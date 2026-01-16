import numpy as np
import sklearn.metrics as skm

# This code consists of top1, top3, top5, accuracy, precision, recall, specificity, f1, auc, and kappa (also confusion matrix)

def topk_accuracy(output:np.ndarray,target:np.ndarray,topk=(1,3,5)):
    ranks = np.argsort(-output,axis=1)
    res={}
    for k in topk:
        correct=0
        topk_preds=ranks[:,:k]
        for i in range(target.shape[0]):
            if target[i] in topk_preds[i]:
                correct+=1
        res[f'top{k}']=correct/target.shape[0]
    return res

def classification_metrics(y_true,y_pred,y_score,average='macro'):
    metrics = {}
    metrics.update(topk_accuracy(y_score,y_true,topk=(1,3,5)))
    metrics['accuracy']=skm.accuracy_score(y_true,y_pred)
    metrics['precision']=skm.precision_score(y_true,y_pred,average=average,zero_division=0)
    metrics['recall']=skm.recall_score(y_true,y_pred,average=average,zero_division=0)
    metrics['f1']=skm.f1_score(y_true,y_pred,average=average,zero_division=0)

    cm=skm.confusion_matrix(y_true,y_pred)
    tn,fp=[],[]
    for i in range(cm.shape[0]):
        tp=cm[i,i]
        fn=cm[i,:].sum()-tp
        fp_i=cm[:,i].sum()-tp
        tn_i=cm.sum()-(tp+fp_i+fn)
        tn.append(tn_i)
        fp.append(fp_i)
    tn=np.array(tn,dtype=float)
    fp=np.array(fp,dtype=float)
    metrics['specificity']=np.nanmean(tn/(tn+fp+1e-8))

    try:
        y_true_bin=skm.label_binarize(y_true,classes=np.arange(y_score.shape[1]))
        aucs=[skm.roc_auc_score(y_true_bin[:,i],y_score[:,i]) for i in range(y_score.shape[1])]
        metrics['auc_macro']=np.nanmean(aucs)
        metrics['auc_per_class']=aucs
    except: metrics['auc_macro']=float('nan');metrics['auc_per_class']=[]

    try: metrics['kappa']=skm.cohen_kappa_score(y_true,y_pred)
    except: metrics['kappa']=float('nan')
    return metrics

def confusion_matrix(y_true,y_pred):
    return skm.confusion_matrix(y_true,y_pred)
