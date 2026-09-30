"""Original synthetic data; latent stochastic labels, never a real fraud dataset."""
import csv
import hashlib
import math
import random
from datetime import datetime,timedelta,timezone
from pathlib import Path
import pandas as pd
from .schemas import LabeledTransaction,Transaction

COLUMNS=list(LabeledTransaction.model_fields)
def generate(rows=3000,seed=20260929,start="2026-01-01T00:00:00+00:00",drift=False,prefix="hist",concept_drift=False):
    rng=random.Random(seed)
    timestamp=datetime.fromisoformat(start)
    if timestamp.tzinfo is None or timestamp.microsecond:
        raise ValueError("Generator start must be timezone aware, whole seconds")
    for i in range(rows):
        timestamp+=timedelta(seconds=rng.randint(15,90))
        account=rng.randrange(40)
        trusted=f"device-{account}"
        mismatch=rng.random()<(.18 if drift else .05)
        amount=min(round(rng.lognormvariate(4.1 if not drift else 5.0,.9),2),900000)
        category=rng.choices(["groceries","electronics","travel","gaming","other"],[40,20,15,10,15])[0]
        country="US" if rng.random()>(.35 if drift else .08) else rng.choice(["GB","IN","DE","CA"])
        latent=-4.2 + 2.2*mismatch + 1.4*(amount>250) + .8*(category=="gaming") + .8*(country!="US")
        if concept_drift:
            latent+=1.2*(category=="travel")
        risk=1/(1+math.exp(-latent))
        yield {"transaction_id":f"{prefix}-{seed}-{i:06d}","account_id":f"account-{account}","timestamp":timestamp.astimezone(timezone.utc).isoformat(),"amount":max(.01,amount),"merchant_id":f"merchant-{rng.randrange(80)}","category":category,"country":country,"currency":"USD","device_id":trusted if not mismatch else f"new-device-{rng.randrange(500)}","trusted_device_id":trusted,"payment_method":rng.choice(["card","wallet","bank"]),"fraud":int(rng.random()<risk)}

def write_generated(path,rows=3000,**kwargs):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=COLUMNS);writer.writeheader();writer.writerows(generate(rows=rows,**kwargs))

def load_events(path,labeled=True):
    path=Path(path)
    frame=pd.read_parquet(path) if path.suffix==".parquet" else pd.read_csv(path)
    schema=LabeledTransaction if labeled else Transaction
    required=set(schema.model_fields)
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    extra=set(frame.columns)-required
    if extra and not (not labeled and extra=={"fraud"}):
        raise ValueError(f"Unexpected columns: {sorted(extra)}")
    validated=[schema.model_validate({key:row[key] for key in required}).model_dump(mode="json") for row in frame.to_dict("records")]
    cleaned=pd.DataFrame(validated)
    if cleaned.empty or cleaned.transaction_id.duplicated().any():
        raise ValueError("Dataset must be nonempty with unique transaction IDs")
    cleaned["timestamp"]=pd.to_datetime(cleaned.timestamp,utc=True)
    if (cleaned.groupby("account_id").currency.nunique()>1).any():
        raise ValueError("Account currency must be stable; FX conversion is not implemented")
    if cleaned.duplicated(["account_id","timestamp"]).any():
        raise ValueError("Per-account event timestamps must be unique")
    return cleaned.sort_values(["timestamp","transaction_id"]).reset_index(drop=True)

def dataset_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
