from collections import deque
from datetime import timezone
import math
import numpy as np
import pandas as pd
from .schemas import Transaction

FEATURE_VERSION="riskflow-prior-events-v1"
RISK={"groceries":.1,"electronics":.6,"travel":.5,"gaming":.8,"other":.3}
CORE=["amount","hour","device_mismatch","merchant_risk"]
NUMERIC=CORE+["count_5m","count_1h","count_24h","spend_1h","mean_24h","std_24h","unique_merchants_1h","time_since_previous","device_changed","country_changed","amount_zscore"]
CATEGORICAL=["category","country","payment_method","currency"]
FEATURES=NUMERIC+CATEGORICAL
FORBIDDEN={"fraud","transaction_id","account_id","timestamp","device_id","trusted_device_id","merchant_id"}

def seconds(event):
    return int(event.timestamp.timestamp())
def features_for(event,prior):
    """Only this account's strictly earlier events in [t-24h,t)."""
    now=seconds(event)
    history=[p for p in prior if p.account_id==event.account_id and now-86400<=seconds(p)<now]
    hour=[p for p in history if seconds(p)>=now-3600]
    amounts=[p.amount for p in history]
    mean=float(np.mean(amounts)) if amounts else 0.0
    std=float(np.std(amounts,ddof=0)) if amounts else 0.0
    last=max(history,key=seconds) if history else None
    return {"amount":event.amount,"hour":event.timestamp.astimezone(timezone.utc).hour,"device_mismatch":int(event.device_id!=event.trusted_device_id),"merchant_risk":RISK[event.category],"count_5m":sum(seconds(p)>=now-300 for p in history),"count_1h":len(hour),"count_24h":len(history),"spend_1h":sum(p.amount for p in hour),"mean_24h":mean,"std_24h":std,"unique_merchants_1h":len({p.merchant_id for p in hour}),"time_since_previous":now-seconds(last) if last else 86400,"device_changed":int(last is not None and event.device_id!=last.device_id),"country_changed":int(last is not None and event.country!=last.country),"amount_zscore":(event.amount-mean)/std if std>1e-9 else 0.0,"category":event.category,"country":event.country,"payment_method":event.payment_method,"currency":event.currency}
class FeatureState:
    """In-memory equivalent for offline runs; serving persists event history in SQLite."""
    def __init__(self):
        self.history={}
        self.last_seen={}
        self.currency={}
    def add(self,event):
        event=event if isinstance(event,Transaction) else Transaction.model_validate(event)
        ts=seconds(event)
        if event.account_id in self.currency and self.currency[event.account_id]!=event.currency:
            raise ValueError("Account currency changed; FX conversion is not implemented")
        self.currency[event.account_id]=event.currency
        if ts<=self.last_seen.get(event.account_id,-math.inf):
            raise ValueError("Per-account events must be strictly time ordered")
        history=self.history.setdefault(event.account_id,deque())
        while history and seconds(history[0])<ts-86400:
            history.popleft()
        value=features_for(event,history)
        history.append(event);self.last_seen[event.account_id]=ts
        return value

def offline_features(frame):
    state=FeatureState();rows=[]
    ordered=frame.sort_values(["timestamp","transaction_id"])
    for row in ordered.to_dict("records"):
        payload={key:row[key] for key in Transaction.model_fields}
        event=Transaction.model_validate(payload)
        features=state.add(event)
        rows.append({**row,**features,"feature_version":FEATURE_VERSION})
    return pd.DataFrame(rows)

def feature_columns(feature_set="full"):
    return (CORE if feature_set=="core" else NUMERIC)+CATEGORICAL

def validate_feature_frame(frame,columns):
    if set(columns)&FORBIDDEN:
        raise ValueError("Label/identifier/time leakage in model inputs")
    if any(column not in frame for column in columns):
        raise ValueError("Required feature column missing")
    numbers=[column for column in columns if column in NUMERIC]
    if not np.isfinite(frame[numbers].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite numeric features")
    if frame[columns].isna().any().any():
        raise ValueError("Missing model features")
