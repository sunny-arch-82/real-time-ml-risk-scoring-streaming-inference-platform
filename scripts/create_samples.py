"""Writes fixtures only; no training, features, inference, Spark or Kafka execution."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from riskflow.data import write_generated

def main():
    directory=ROOT/"data"/"sample";directory.mkdir(parents=True,exist_ok=True)
    write_generated(directory/"transactions.csv",3000)
    write_generated(directory/"live_transactions.csv",100,seed=20260930,start="2026-02-01T00:00:00+00:00",prefix="live")
    write_generated(directory/"drift_transactions.csv",500,seed=20261001,start="2026-02-02T00:00:00+00:00",drift=True,prefix="drift")
    import json
    with (directory/"live_transactions.csv").open() as file:row=next(csv.DictReader(file))
    row.pop("fraud");row["amount"]=float(row["amount"])
    (directory/"request.json").write_text(json.dumps(row,indent=2))
if __name__=="__main__":main()
