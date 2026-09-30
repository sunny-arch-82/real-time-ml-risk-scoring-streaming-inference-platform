import pandas as pd
import pytest
from riskflow.data import generate,load_events
from riskflow.features import FeatureState,offline_features,validate_feature_frame,FEATURES
from riskflow.schemas import Transaction

def event(i,seconds=60,amount=10):
    return Transaction(transaction_id=f"T-{i}",account_id="A",timestamp=f"2026-01-01T00:{seconds//60:02d}:{seconds%60:02d}+00:00",amount=amount,merchant_id="M",category="groceries",country="US",currency="USD",device_id="D",trusted_device_id="D",payment_method="card")
def test_deterministic_generation():assert list(generate(20,seed=7))==list(generate(20,seed=7))
def test_schema_rejects_bad_event():
    with pytest.raises(ValueError):Transaction.model_validate({**event(1).model_dump(),"amount":-1})
    with pytest.raises(ValueError):Transaction.model_validate({**event(1).model_dump(),"timestamp":"2026-01-01T00:00:00"})
def test_prior_only_consistency():
    events=[event(1,60,10),event(2,120,30),event(3,180,50)]
    state=FeatureState();online=[state.add(e) for e in events]
    frame=offline_features(pd.DataFrame([e.model_dump() for e in events]))
    assert online[0]["count_1h"]==0
    assert online[2]["spend_1h"]==40 and online[2]["mean_24h"]==20
    for i,features in enumerate(online):
        for column in FEATURES:assert frame.iloc[i][column]==features[column]
def test_leakage_and_late_events():
    state=FeatureState();state.add(event(1))
    with pytest.raises(ValueError):state.add(event(2))
    with pytest.raises(ValueError):validate_feature_frame(pd.DataFrame({"fraud":[1]}),["fraud"])
def test_dataset_duplicates(tmp_path):
    frame=pd.DataFrame(list(generate(10)));frame=pd.concat([frame,frame.iloc[[0]]])
    path=tmp_path/"x.csv";frame.to_csv(path,index=False)
    with pytest.raises(ValueError):load_events(path)
