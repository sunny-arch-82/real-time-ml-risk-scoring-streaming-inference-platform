"""Tiny training/inference smoke: explicit local/CI opt-in, not run at generation."""
import json
import pandas as pd
from riskflow.data import generate
from riskflow.config import TrainConfig,Policy
from riskflow.training import train
from riskflow.registry import Registry
from riskflow.batch import batch_score
from riskflow.serving import Predictor
from riskflow.schemas import Transaction

def test_small_lifecycle(tmp_path,monkeypatch):
    monkeypatch.setenv("RISKFLOW_HOME",str(tmp_path/"artifacts"))
    monkeypatch.setenv("RISKFLOW_REGISTRY",str(tmp_path/"models"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI",f"sqlite:///{tmp_path/'mlflow.db'}")
    frame=pd.DataFrame(list(generate(500,seed=21)))
    # Force tiny fixture both classes; fixture smoke is not a quality benchmark.
    frame["fraud"]=[i%7==0 for i in range(len(frame))];frame["fraud"]=frame.fraud.astype(int)
    path=tmp_path/"transactions.csv";frame.to_csv(path,index=False)
    report=train(path,TrainConfig(models=["logistic"],calibration="none"))
    version=report["recommended_candidate"]
    registry=Registry();result=registry.promote(version,Policy(minimum_validation_ap=0,allowed_ap_drop=1))
    assert result["promoted"]
    scored=batch_score(path,tmp_path/"batch.csv")
    assert len(scored)==500 and scored.probability.between(0,1).all()
    predictor=Predictor()
    event=Transaction.model_validate({key:frame.iloc[0][key] for key in Transaction.model_fields})
    first=predictor.predict(event);second=predictor.predict(event)
    assert first.model_version==version and second.duplicate
    assert first.features["count_24h"]==0
    assert json.loads((tmp_path/"models/current.json").read_text())["version"]==version
