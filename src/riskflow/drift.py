import numpy as np
from scipy.spatial.distance import jensenshannon
from .features import NUMERIC,CATEGORICAL
from .evaluation import classification_metrics

def reference_profile(frame):
    reference={"rows":len(frame),"numeric":{},"categorical":{}}
    for column in NUMERIC:
        values=frame[column].dropna().to_numpy(dtype=float)
        cuts=np.unique(np.quantile(values,np.linspace(0,1,11)[1:-1])).tolist()
        histogram=np.histogram(values,bins=[-np.inf,*cuts,np.inf])[0]
        reference["numeric"][column]={"cuts":cuts,"counts":histogram.tolist()}
    for column in CATEGORICAL:
        reference["categorical"][column]=frame[column].fillna("__missing__").astype(str).value_counts().to_dict()
    return reference

def psi(reference,current):
    p=np.asarray(reference,dtype=float)+1e-6;q=np.asarray(current,dtype=float)+1e-6
    p/=p.sum();q/=q.sum()
    return float(np.sum((q-p)*np.log(q/p)))

def analyze_drift(reference,recent,policy,threshold):
    if recent.empty:raise ValueError("Recent monitoring dataset is empty")
    numeric,categorical,missing={},{},{}
    for column,profile in reference["numeric"].items():
        if column not in recent:
            missing[column]=1.0;continue
        values=recent[column].dropna().to_numpy(dtype=float)
        missing[column]=float(recent[column].isna().mean())
        histogram=np.histogram(values,bins=[-np.inf,*profile["cuts"],np.inf])[0]
        numeric[column]=psi(profile["counts"],histogram) if len(values) else None
    for column,counts in reference["categorical"].items():
        if column not in recent:
            missing[column]=1.0;continue
        missing[column]=float(recent[column].isna().mean())
        current=recent[column].fillna("__missing__").astype(str).value_counts().to_dict()
        keys=sorted(set(counts)|set(current))
        p=np.array([counts.get(key,0)+1e-6 for key in keys]);q=np.array([current.get(key,0)+1e-6 for key in keys])
        categorical[column]=float(jensenshannon(p/p.sum(),q/q.sum(),base=2)**2)
    enough=len(recent)>=policy.minimum_monitoring_rows
    drifted=[key for key,value in numeric.items() if value is not None and value>=policy.psi_threshold]+[key for key,value in categorical.items() if value>=policy.js_threshold]
    monitoring={"rows":len(recent),"positive_prediction_rate":float(recent.predicted_label.mean()) if "predicted_label" in recent else None,"mean_probability":float(recent.probability.mean()) if "probability" in recent else None,"latency_ms_mean":float(recent.latency_ms.mean()) if "latency_ms" in recent else None}
    prediction_drift=None
    if "probability" in recent and reference.get("prediction"):
        profile=reference["prediction"]
        prediction_drift=psi(profile["counts"],np.histogram(recent.probability.dropna(),bins=[-np.inf,*profile["cuts"],np.inf])[0])
    performance=None;labels=0
    if {"fraud","probability"}<=set(recent.columns):
        labeled=recent.dropna(subset=["fraud","probability"]);labels=len(labeled)
        if labels:performance=classification_metrics(labeled.fraud,labeled.probability,threshold)
    return {"data_drift":{"numeric_psi":numeric,"categorical_js_divergence":categorical,"flagged_features":drifted,"sufficient_rows":enough},"prediction_drift_psi":prediction_drift,"performance":performance,"labeled_rows":labels,"missing_rates":missing,"monitoring":monitoring,"policy":policy.model_dump(),"note":"Example heuristics; data/prediction drift alone do not prove model deterioration"}

def retraining_decision(report,policy,new_label_count=0,scheduled=False):
    enough=new_label_count>=policy.minimum_new_labels
    data_drift=report["data_drift"]["sufficient_rows"] and bool(report["data_drift"]["flagged_features"])
    ap=(report.get("performance") or {}).get("pr_auc")
    performance_drop=report.get("labeled_rows",0)>=policy.minimum_new_labels and ap is not None and ap<policy.minimum_recent_ap
    reasons=[]
    if data_drift:reasons.append("Feature distribution drift detected")
    if performance_drop:reasons.append("Recent labeled AP below configured minimum")
    if scheduled:reasons.append("Scheduled review requested")
    recommend=enough and bool(reasons)
    return {"recommend_retraining":recommend,"new_label_count":new_label_count,"enough_new_labels":enough,"reasons":reasons,"action":"Train a candidate and evaluate before manual promotion" if recommend else "Collect labels/investigate; retain champion","automatic_training":False}
