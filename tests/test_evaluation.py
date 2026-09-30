import numpy as np
import pandas as pd
import pytest
from riskflow.config import ThresholdConfig,TrainConfig,Policy
from riskflow.evaluation import choose_threshold,classification_metrics
from riskflow.calibration import ProbabilityCalibration
from riskflow.training import temporal_segments
from riskflow.drift import psi,retraining_decision

def test_threshold_and_metrics():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.7,.8])
    threshold,_=choose_threshold(y,p,ThresholdConfig())
    assert classification_metrics(y,p,threshold)["f1"]==1
    with pytest.raises(ValueError):choose_threshold([0,0],[.1,.2],ThresholdConfig())
def test_calibration():
    p=np.array([.1,.2,.3,.7,.8,.9]);y=np.array([0,0,0,1,1,1])
    for method in ("none","sigmoid","isotonic"):
        values=ProbabilityCalibration(method).fit(p,y).predict(p)
        assert np.all((values>=0)&(values<=1))
def test_temporal_split_no_overlap():
    frame=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=200,freq="min",tz="UTC"),"fraud":[i%2 for i in range(200)]})
    parts=temporal_segments(frame,TrainConfig())
    assert sum(map(len,parts))==200
    assert all(a.timestamp.max()<b.timestamp.min() for a,b in zip(parts,parts[1:]))
def test_drift_not_automatic_retrain():
    assert psi([10,20],[10,20])==pytest.approx(0)
    report={"data_drift":{"sufficient_rows":True,"flagged_features":["amount"]},"performance":None,"labeled_rows":0}
    assert not retraining_decision(report,Policy(),new_label_count=0)["recommend_retraining"]
    assert retraining_decision(report,Policy(),new_label_count=200)["recommend_retraining"]
