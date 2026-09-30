"""Local request telemetry, no payloads or credentials recorded."""
import os
from pathlib import Path
import sqlite3
from datetime import datetime,timezone
import numpy as np

class RequestMonitor:
    def __init__(self):
        home=Path(os.getenv("RISKFLOW_HOME","artifacts"));home.mkdir(parents=True,exist_ok=True)
        self.path=home/"requests.db"
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, timestamp TEXT, path TEXT, status INTEGER, latency_ms REAL)")
    def record(self,path,status,latency_ms):
        with sqlite3.connect(self.path,timeout=30) as db:
            db.execute("INSERT INTO requests(timestamp,path,status,latency_ms) VALUES (?,?,?,?)",(datetime.now(timezone.utc).isoformat(),path,status,latency_ms))
    def summary(self):
        with sqlite3.connect(self.path) as db:
            records=db.execute("SELECT status,latency_ms FROM requests WHERE path='/predict' ORDER BY id DESC LIMIT 1000").fetchall()
        latencies=[r[1] for r in records]
        return {"request_window":len(records),"error_rate":sum(r[0]>=400 for r in records)/len(records) if records else None,"latency_ms_mean":float(np.mean(latencies)) if latencies else None,"latency_ms_p95":float(np.percentile(latencies,95)) if len(records)>=20 else None,"note":"Predict requests only; includes duplicates and rejected requests, process telemetry not a load benchmark"}
