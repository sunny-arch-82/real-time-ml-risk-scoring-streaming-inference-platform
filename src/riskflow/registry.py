import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import joblib
from .features import FEATURE_VERSION

class Registry:
    """Local complete-bundle lifecycle, paired with MLflow raw-model versions."""
    def __init__(self,path=None):
        self.path=Path(path or os.getenv("RISKFLOW_REGISTRY","models"));self.path.mkdir(parents=True,exist_ok=True)
    @contextlib.contextmanager
    def lock(self):
        with (self.path/".registry.lock").open("a") as file:
            fcntl.flock(file,fcntl.LOCK_EX)
            try:yield
            finally:fcntl.flock(file,fcntl.LOCK_UN)
    def directory(self,version):
        if not re.fullmatch(r"[a-f0-9]{32}",version):raise ValueError("Invalid model version")
        return self.path/version
    def current(self):
        file=self.path/"current.json"
        return json.loads(file.read_text()) if file.exists() else None
    def metadata(self,version):
        return json.loads((self.directory(version)/"metadata.json").read_text())
    def save(self,version,bundle,metadata):
        with self.lock():
            directory=self.directory(version);directory.mkdir()
            path=directory/"bundle.joblib";joblib.dump(bundle,path)
            metadata={**metadata,"bundle_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            (directory/"metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False))
        return metadata
    def load(self,version=None):
        pointer=self.current()
        version=version or (pointer["version"] if pointer else None)
        if version is None:raise FileNotFoundError("No champion selected; train and explicitly promote a candidate")
        metadata=self.metadata(version)
        if metadata["feature_version"]!=FEATURE_VERSION:raise ValueError("Feature contract mismatch")
        path=self.directory(version)/"bundle.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata["bundle_sha256"]:raise ValueError("Model checksum mismatch")
        # Only locally produced trusted artifacts; joblib deserialization executes code.
        return joblib.load(path),metadata
    def promote(self,version,policy,rollback=False):
        with self.lock():
            candidate=self.metadata(version);current=self.current()
            reasons=[]
            if not rollback:
                metrics=candidate["validation_metrics"]
                if metrics["pr_auc"] is None or metrics["pr_auc"]<policy.minimum_validation_ap:reasons.append("Candidate validation AP below policy")
                if metrics["precision"]<policy.minimum_precision:reasons.append("Candidate precision below policy")
                if metrics["recall"]<policy.minimum_recall:reasons.append("Candidate recall below policy")
                if current:
                    champion=self.metadata(current["version"])
                    if candidate["validation_hash"]!=champion["validation_hash"]:reasons.append("Validation datasets differ; re-evaluate both on one common holdout")
                    elif metrics["pr_auc"]<champion["validation_metrics"]["pr_auc"]-policy.allowed_ap_drop:reasons.append("Candidate AP regresses beyond allowed margin")
            elif not current or version not in current.get("history",[]):
                reasons.append("Rollback must select a previously promoted version")
            if reasons:return {"promoted":False,"reasons":reasons}
            self.load(version)
            history=list(dict.fromkeys([*(current.get("history",[]) if current else []),*( [current["version"]] if current else []),version]))
            pointer={"version":version,"previous_version":current["version"] if current else None,"history":history}
            # MLflow aliases are bookkeeping; the atomic complete-bundle pointer is authoritative.
            import mlflow
            mlflow.set_tracking_uri(candidate["tracking_uri"])
            mlflow.tracking.MlflowClient().set_registered_model_alias("RiskFlow","champion",candidate["mlflow_model_version"])
            temp=self.path/"current.tmp";temp.write_text(json.dumps(pointer,indent=2));temp.replace(self.path/"current.json")
            return {"promoted":True,"version":version,"rollback":rollback,"previous_version":pointer["previous_version"]}
