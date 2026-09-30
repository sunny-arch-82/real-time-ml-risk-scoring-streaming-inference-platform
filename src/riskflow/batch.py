from pathlib import Path
import pandas as pd
from .data import load_events
from .features import offline_features
from .registry import Registry
from .training import score_bundle

def batch_score(input_path,output_path,version=None,registry_path=None):
    raw=load_events(input_path,labeled=False)
    frame=offline_features(raw)
    bundle,metadata=Registry(registry_path).load(version)
    probability=score_bundle(bundle,frame)
    frame["probability"]=probability;frame["predicted_label"]=(probability>=bundle["threshold"]).astype(int)
    frame["threshold"]=bundle["threshold"];frame["model_version"]=metadata["version"]
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix==".parquet":frame.to_parquet(output,index=False)
    else:frame.to_csv(output,index=False)
    return frame

def export_monitoring(state_path,output_path):
    import json,sqlite3
    rows=[]
    with sqlite3.connect(state_path) as db:
        for payload,prediction in db.execute("SELECT payload,prediction FROM events WHERE prediction IS NOT NULL ORDER BY event_second"):
            event=json.loads(payload);prediction=json.loads(prediction)
            rows.append({**event,**prediction["features"],**{key:value for key,value in prediction.items() if key!="features"}})
    frame=pd.DataFrame(rows)
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False);return frame
