import os
from pathlib import Path
from threading import Lock
import time
import logging
from .monitoring import RequestMonitor
from fastapi import FastAPI,HTTPException
from .schemas import Transaction,Prediction
from .serving import Predictor
from .registry import Registry

app=FastAPI(title="RiskFlow",version="0.1.0",description="Synthetic transaction risk prototype. No financial decisions or authentication.")
service=None;service_lock=Lock()
@app.middleware("http")
async def request_telemetry(request,call_next):
    started=time.perf_counter();status=500
    try:
        response=await call_next(request);status=response.status_code;return response
    finally:
        if request.url.path=="/predict":
            try:RequestMonitor().record("/predict",status,(time.perf_counter()-started)*1000)
            except Exception:
                logging.getLogger("riskflow").exception("Request telemetry write failed")
def predictor():
    global service
    with service_lock:
        if service is None:service=Predictor()
    return service
@app.get("/health")
def health():return {"status":"ok","domain":"synthetic_only"}
@app.get("/ready")
def ready():
    try:
        if not predictor().ready():raise HTTPException(503,"Train and explicitly promote a candidate first")
        return {"ready":True}
    except (ValueError,OSError) as exc:raise HTTPException(503,str(exc)) from exc
@app.get("/model-info")
def model_info():
    current=Registry().current()
    if not current:raise HTTPException(503,"No promoted model")
    metadata=Registry().metadata(current["version"])
    return {key:metadata[key] for key in ("version","model_type","feature_version","feature_set","threshold","calibration","dataset_hash","mlflow_model_version")}
@app.post("/predict",response_model=Prediction)
def predict(event:Transaction):
    try:return predictor().predict(event)
    except FileNotFoundError as exc:raise HTTPException(503,str(exc)) from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
@app.get("/monitoring")
def monitoring():
    path=Path(os.getenv("RISKFLOW_HOME","artifacts"))/"online-state.db"
    if not path.exists():return {"predictions":0,"requests":RequestMonitor().summary(),"note":"No predictions yet"}
    import sqlite3,json
    with sqlite3.connect(path) as db:
        predictions=[json.loads(row[0]) for row in db.execute("SELECT prediction FROM events WHERE prediction IS NOT NULL ORDER BY event_second DESC LIMIT 1000")]
    return {"window_rows":len(predictions),"mean_probability":sum(p["probability"] for p in predictions)/len(predictions) if predictions else None,"positive_rate":sum(p["predicted_label"] for p in predictions)/len(predictions) if predictions else None,"mean_latency_ms":sum(p["latency_ms"] for p in predictions)/len(predictions) if predictions else None,"requests":RequestMonitor().summary(),"note":"Prediction distribution uses successful unique events; separate request statistics include failures"}
