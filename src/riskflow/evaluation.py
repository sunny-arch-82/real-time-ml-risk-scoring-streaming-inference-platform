import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,precision_recall_curve,brier_score_loss,log_loss
from sklearn.calibration import calibration_curve

def classification_metrics(y,probability,threshold):
    y=np.asarray(y);p=np.asarray(probability,dtype=float)
    if not np.isin(y,[0,1]).all():raise ValueError("Labels must be binary 0/1")
    y=y.astype(int)
    if not len(y) or len(y)!=len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Nonempty labels/probabilities of equal length in [0,1] required")
    pred=(p>=threshold).astype(int)
    both=len(np.unique(y))==2
    precision,recall,cutoffs=precision_recall_curve(y,p)
    fraction,mean=calibration_curve(y,p,n_bins=10,strategy="quantile")
    return {"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,p)) if both else None,"pr_auc":float(average_precision_score(y,p)) if both else None,"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),"brier_score":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"positive_prediction_rate":float(pred.mean()),"threshold":float(threshold),"samples":len(y),"positive_labels":int(y.sum()),"precision_recall_curve":{"precision":precision.tolist(),"recall":recall.tolist(),"thresholds":cutoffs.tolist()},"calibration_curve":{"fraction_positive":fraction.tolist(),"mean_probability":mean.tolist()}}

def choose_threshold(y,p,config):
    y=np.asarray(y);p=np.asarray(p)
    if len(np.unique(y))!=2:
        raise ValueError("Threshold segment requires both label classes")
    if config.objective=="top_k":
        threshold=float(np.sort(p)[-min(config.top_k,len(p))])
        return threshold,{"objective":"top_k","requested_k":config.top_k,"selected_count":int((p>=threshold).sum()),"note":"Tied scores can exceed capacity; no exact-capacity guarantee"}
    candidates=np.unique(np.r_[0,p,1])
    choices=[]
    for threshold in candidates:
        prediction=p>=threshold
        tp=int(((y==1)&prediction).sum());fp=int(((y==0)&prediction).sum());fn=int(((y==1)&~prediction).sum())
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        if config.objective=="minimum_recall" and recall<config.minimum_recall:
            continue
        if config.objective=="minimum_precision" and precision<config.minimum_precision:
            continue
        objective={"f1":f1,"minimum_recall":precision,"minimum_precision":recall,"cost":-(fp*config.false_positive_cost+fn*config.false_negative_cost)}[config.objective]
        choices.append((objective,float(threshold),precision,recall))
    if not choices:
        raise ValueError("No threshold meets the configured objective constraint")
    best=max(choices,key=lambda item:(item[0],item[1]))
    return best[1],{"objective":config.objective,"objective_value":best[0],"precision":best[2],"recall":best[3],"selected_on":"threshold_validation_only"}

def error_analysis(frame,y,p,threshold):
    result=frame.copy();result["probability"]=p;result["prediction"]=(np.asarray(p)>=threshold).astype(int)
    result["error_type"]=np.where((np.asarray(y)==0)&(result.prediction==1),"false_positive",np.where((np.asarray(y)==1)&(result.prediction==0),"false_negative","correct"))
    result["high_confidence_error"]=((result.error_type=="false_positive")&(result.probability>=.9))|((result.error_type=="false_negative")&(result.probability<=.1))
    result["amount_range"]=pd.cut(result.amount,[0,50,200,1000,float("inf")],labels=["0-50","50-200","200-1000","1000+"]).astype(str)
    result["time_day"]=pd.to_datetime(result.timestamp,utc=True).dt.strftime("%Y-%m-%d")
    breakdown={}
    for column in ("category","country","payment_method","device_mismatch","amount_range","time_day"):
        breakdown[column]={str(value):classification_metrics(group.fraud,group.probability,threshold) for value,group in result.groupby(column)}
    return result[result.error_type!="correct"],breakdown
