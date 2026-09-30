from datetime import datetime,timezone
import json
import os
from pathlib import Path
import uuid
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .data import load_events,dataset_hash
from .features import offline_features,feature_columns,validate_feature_frame,FEATURE_VERSION
from .modeling import fit_model
from .calibration import ProbabilityCalibration
from .evaluation import classification_metrics,choose_threshold,error_analysis
from .drift import reference_profile
from .registry import Registry

def temporal_segments(frame,config):
    # Timestamp boundaries: equal-time groups never straddle segments.
    unique=sorted(pd.to_datetime(frame.timestamp,utc=True).unique())
    boundaries=[int(len(unique)*f) for f in (config.train_fraction,config.train_fraction+config.calibration_fraction,config.train_fraction+config.calibration_fraction+config.threshold_fraction)]
    if any(i<=0 or i>=len(unique) for i in boundaries) or len(set(boundaries))!=3:raise ValueError("Too few timestamps for temporal segments")
    cut=[unique[i] for i in boundaries]
    segments=[frame[frame.timestamp<cut[0]],frame[(frame.timestamp>=cut[0])&(frame.timestamp<cut[1])],frame[(frame.timestamp>=cut[1])&(frame.timestamp<cut[2])],frame[frame.timestamp>=cut[2]]]
    for segment in segments:
        if len(segment)<10 or segment.fraud.nunique()<2:raise ValueError("Every train/calibration/threshold/test segment needs >=10 rows and both classes; generate more data")
    return segments

def score_bundle(bundle,frame):
    validate_feature_frame(frame,bundle["columns"])
    probability=bundle["model"].predict_proba(frame[bundle["columns"]])[:,1]
    return bundle["calibrator"].predict(probability)

def train(data_path,config,tune=False,registry_path=None,feature_path=None):
    raw=load_events(data_path)
    if feature_path:
        frame=pd.read_parquet(feature_path)
        frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True)
        frame=frame.sort_values(["timestamp","transaction_id"]).reset_index(drop=True)
        if frame.transaction_id.duplicated().any() or set(frame.transaction_id)!=set(raw.transaction_id):
            raise ValueError("Precomputed feature IDs must match raw dataset exactly")
        if "feature_version" not in frame or not (frame.feature_version==FEATURE_VERSION).all():
            raise ValueError("Precomputed feature contract mismatch")
        left=frame.set_index("transaction_id");right=raw.set_index("transaction_id")
        for column in ("fraud","account_id","timestamp","amount","category","country","currency","payment_method"):
            if not left[column].sort_index().equals(right[column].sort_index()):
                # Spark integer widths are allowed; compare values independent of dtype.
                if not (left[column].sort_index().to_numpy()==right[column].sort_index().to_numpy()).all():
                    raise ValueError(f"Precomputed raw column mismatch: {column}")
    else:
        frame=offline_features(raw)
    train_frame,cal_frame,val_frame,test_frame=temporal_segments(frame,config)
    columns=feature_columns(config.feature_set);validate_feature_frame(frame,columns)
    home=Path(os.getenv("RISKFLOW_HOME","artifacts"));home.mkdir(parents=True,exist_ok=True)
    tracking=os.getenv("MLFLOW_TRACKING_URI",f"sqlite:///{(home/'mlflow.db').resolve()}")
    mlflow.set_tracking_uri(tracking)
    if mlflow.get_experiment_by_name("riskflow-synthetic") is None:
        mlflow.create_experiment("riskflow-synthetic",artifact_location=(home/"mlflow-artifacts").resolve().as_uri())
    mlflow.set_experiment("riskflow-synthetic")
    registry=Registry(registry_path);results=[]
    for kind in config.models:
        version=uuid.uuid4().hex;directory=home/"training"/version;directory.mkdir(parents=True)
        with mlflow.start_run(run_name=f"{kind}-{version[:8]}") as run:
            model,search=fit_model(kind,train_frame[columns],train_frame.fraud,columns,config,tune)
            calibration=ProbabilityCalibration(config.calibration).fit(model.predict_proba(cal_frame[columns])[:,1],cal_frame.fraud)
            bundle={"model":model,"calibrator":calibration,"columns":columns,"feature_version":FEATURE_VERSION}
            validation_p=score_bundle(bundle,val_frame)
            threshold,selection=choose_threshold(val_frame.fraud,validation_p,config.threshold)
            bundle["threshold"]=threshold
            metrics=classification_metrics(val_frame.fraud,validation_p,threshold)
            test_p=score_bundle(bundle,test_frame)
            test_metrics=classification_metrics(test_frame.fraud,test_p,threshold)
            raw_test=classification_metrics(test_frame.fraud,model.predict_proba(test_frame[columns])[:,1],threshold)
            errors,breakdown=error_analysis(test_frame,test_frame.fraud,test_p,threshold)
            errors.to_csv(directory/"errors.csv",index=False)
            # Permutation importance on threshold validation, not final test.
            importance=permutation_importance(model,val_frame[columns],val_frame.fraud,scoring="average_precision",n_repeats=3,random_state=config.seed,n_jobs=1)
            importance_report=dict(zip(columns,importance.importances_mean.tolist()))
            reference=reference_profile(train_frame)
            ref_probability=score_bundle(bundle,train_frame)
            cuts=np.linspace(.1,.9,9).tolist();reference["prediction"]={"cuts":cuts,"counts":np.histogram(ref_probability,bins=[-np.inf,*cuts,np.inf])[0].tolist()}
            periods={name:{"start":group.timestamp.min().isoformat(),"end":group.timestamp.max().isoformat(),"rows":len(group),"positives":int(group.fraud.sum())} for name,group in zip(["train","calibration","threshold_validation","test"],[train_frame,cal_frame,val_frame,test_frame])}
            manifest={"version":version,"timestamp":datetime.now(timezone.utc).isoformat(),"model_type":kind,"dataset_hash":dataset_hash(raw),"dataset_version":config.dataset_version,"validation_hash":dataset_hash(val_frame[list(raw.columns)]),"feature_version":FEATURE_VERSION,"feature_set":config.feature_set,"feature_source":str(feature_path) if feature_path else "shared_python", "columns":columns,"seed":config.seed,"threshold":threshold,"calibration":config.calibration,"periods":periods,"validation_metrics":metrics,"test_metrics":test_metrics,"raw_test_metrics":raw_test,"threshold_selection":selection,"search":search,"config":config.model_dump(),"tracking_uri":tracking,"mlflow_run_id":run.info.run_id,"prediction_reference_note":"Training scores, useful distribution reference; optimistic for assessing quality"}
            (directory/"evaluation.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
            (directory/"error_breakdown.json").write_text(json.dumps(breakdown,indent=2,allow_nan=False))
            (directory/"importance.json").write_text(json.dumps(importance_report,indent=2))
            (directory/"reference.json").write_text(json.dumps(reference,indent=2))
            mlflow.log_params({"model_type":kind,"seed":config.seed,"feature_version":FEATURE_VERSION,"feature_set":config.feature_set,"calibration":config.calibration,"threshold":threshold,"tuned":tune})
            mlflow.set_tags({"dataset_hash":manifest["dataset_hash"],"validation_hash":manifest["validation_hash"],"synthetic":"true"})
            for prefix,report in (("validation",metrics),("test",test_metrics)):
                mlflow.log_metrics({prefix+"_"+key:value for key,value in report.items() if key in {"precision","recall","f1","roc_auc","pr_auc","brier_score","log_loss"} and value is not None})
            mlflow.log_artifacts(str(directory),artifact_path="reports")
            mlflow.sklearn.log_model(model,artifact_path="raw_model",input_example=train_frame[columns].head(2))
            registered=mlflow.register_model(f"runs:/{run.info.run_id}/raw_model","RiskFlow")
            manifest["mlflow_model_version"]=str(registered.version)
            manifest["reference"]=reference
            registry.save(version,bundle,manifest)
            mlflow.log_artifact(str(registry.directory(version)/"bundle.joblib"),artifact_path="complete_bundle")
            mlflow.tracking.MlflowClient().set_registered_model_alias("RiskFlow","candidate",registered.version)
            results.append({"version":version,"model_type":kind,"validation_pr_auc":metrics["pr_auc"],"validation_f1":metrics["f1"],"test_pr_auc":test_metrics["pr_auc"],"threshold":threshold})
    recommended=max(results,key=lambda result:result["validation_pr_auc"])
    report={"candidates":results,"recommended_candidate":recommended["version"],"selection_basis":"threshold-validation AP only; final test is descriptive, never automatic promotion","champion_unchanged":True}
    path=home/"training"/f"comparison-{uuid.uuid4().hex[:8]}.json";path.write_text(json.dumps(report,indent=2));return report
