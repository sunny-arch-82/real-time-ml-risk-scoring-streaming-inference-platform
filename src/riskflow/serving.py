import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import pandas as pd
from .schemas import Transaction,Prediction
from .features import features_for,seconds,FEATURE_VERSION
from .training import score_bundle
from .registry import Registry

class StateStore:
    def __init__(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);self.path=path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS events (transaction_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, event_second INTEGER NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL, prediction TEXT)")
            db.execute("CREATE INDEX IF NOT EXISTS account_time ON events(account_id,event_second)")
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.execute("PRAGMA journal_mode=WAL");return db
    def seed(self,frame):
        # Warm-start history is authoritative history, never prediction data.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in frame.to_dict("records"):
                event=Transaction.model_validate({key:row[key] for key in Transaction.model_fields})
                payload=event.model_dump_json();digest=hashlib.sha256(payload.encode()).hexdigest()
                previous=db.execute("SELECT payload_hash FROM events WHERE transaction_id=?",(event.transaction_id,)).fetchone()
                if previous and previous[0]!=digest:raise ValueError("Conflicting warm-start event ID")
                db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,NULL)",(event.transaction_id,event.account_id,seconds(event),payload,digest))
class Predictor:
    def __init__(self,registry_path=None,state_path=None):
        self.registry=Registry(registry_path)
        home=Path(os.getenv("RISKFLOW_HOME","artifacts"))
        self.state=StateStore(state_path or home/"online-state.db")
        self.bundle=None;self.metadata=None;self.lock=threading.RLock()
    def ready(self):
        with self.lock:
            pointer=self.registry.current()
            if not pointer:return False
            if self.metadata is None or self.metadata["version"]!=pointer["version"]:
                self.bundle,self.metadata=self.registry.load(pointer["version"])
            return True
    def predict(self,event):
        started=time.perf_counter()
        with self.lock:
            if not self.ready():raise FileNotFoundError("No promoted model")
            payload=event.model_dump_json();digest=hashlib.sha256(payload.encode()).hexdigest()
            with self.state.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                previous=db.execute("SELECT payload_hash,prediction FROM events WHERE transaction_id=?",(event.transaction_id,)).fetchone()
                if previous:
                    if previous[0]!=digest:raise ValueError("Transaction ID reused with different payload")
                    if previous[1]:
                        cached=Prediction.model_validate_json(previous[1]);cached.duplicate=True;return cached
                    raise ValueError("Transaction is historical warm-start data; send a new later event")
                last_row=db.execute("SELECT event_second,payload FROM events WHERE account_id=? ORDER BY event_second DESC LIMIT 1",(event.account_id,)).fetchone()
                latest=last_row[0] if last_row else None
                if last_row and Transaction.model_validate_json(last_row[1]).currency!=event.currency:
                    raise ValueError("Account currency changed; FX conversion is not implemented")
                if latest is not None and seconds(event)<=latest:raise ValueError("Late/out-of-order event: account timestamps must strictly increase")
                rows=db.execute("SELECT payload FROM events WHERE account_id=? AND event_second>=? ORDER BY event_second",(event.account_id,seconds(event)-86400)).fetchall()
                prior=[Transaction.model_validate_json(row[0]) for row in rows]
                features=features_for(event,prior)
                probability=float(score_bundle(self.bundle,pd.DataFrame([features]))[0]);threshold=self.bundle["threshold"]
                result=Prediction(transaction_id=event.transaction_id,probability=probability,predicted_label=int(probability>=threshold),threshold=threshold,model_version=self.metadata["version"],feature_version=FEATURE_VERSION,features=features,latency_ms=(time.perf_counter()-started)*1000)
                db.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",(event.transaction_id,event.account_id,seconds(event),payload,digest,result.model_dump_json()))
                return result
