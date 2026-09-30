from typing import Literal
from pathlib import Path
import yaml
from pydantic import Field,model_validator
from .schemas import Contract

class ThresholdConfig(Contract):
    objective: Literal["f1","minimum_recall","minimum_precision","cost","top_k"] = "f1"
    minimum_recall: float = Field(default=.7,ge=0,le=1)
    minimum_precision: float = Field(default=.5,ge=0,le=1)
    false_positive_cost: float = Field(default=1,ge=0)
    false_negative_cost: float = Field(default=10,ge=0)
    top_k: int = Field(default=25,ge=1)
class TrainConfig(Contract):
    dataset_version: str = "synthetic-v1"
    seed: int = 20260929
    train_fraction: float = Field(default=.60,gt=0,lt=1)
    calibration_fraction: float = Field(default=.15,gt=0,lt=1)
    threshold_fraction: float = Field(default=.10,gt=0,lt=1)
    models: list[Literal["logistic","random_forest"]] = Field(default_factory=lambda:["logistic","random_forest"])
    feature_set: Literal["core","full"] = "full"
    calibration: Literal["none","sigmoid","isotonic"] = "sigmoid"
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    search_iterations: int = Field(default=6,ge=1,le=200)
    cv_splits: int = Field(default=3,ge=2,le=10)
    cv_gap: int = Field(default=0,ge=0)
    tuning_space: dict[str,dict[str,list]] = Field(default_factory=lambda:{"logistic":{"model__C":[.1,1,10]},"random_forest":{"model__n_estimators":[60,100],"model__max_depth":[5,10,None],"model__min_samples_leaf":[2,5,10]}})
    @model_validator(mode="after")
    def valid_split(self):
        if self.train_fraction+self.calibration_fraction+self.threshold_fraction>=1:
            raise ValueError("Fractions must leave a final test segment")
        if not self.models or len(self.models)!=len(set(self.models)):
            raise ValueError("Choose unique nonempty model types")
        return self
class Policy(Contract):
    # Example policy, never an industry standard or automatically tuned threshold.
    minimum_validation_ap: float = Field(default=.1,ge=0,le=1)
    allowed_ap_drop: float = Field(default=.02,ge=0,le=1)
    minimum_precision: float = Field(default=0,ge=0,le=1)
    minimum_recall: float = Field(default=0,ge=0,le=1)
    psi_threshold: float = Field(default=.2,ge=0)
    js_threshold: float = Field(default=.1,ge=0)
    minimum_recent_ap: float = Field(default=.1,ge=0,le=1)
    minimum_new_labels: int = Field(default=200,ge=1)
    minimum_monitoring_rows: int = Field(default=100,ge=1)

def load_config(path,model=TrainConfig):
    return model.model_validate(yaml.safe_load(Path(path).read_text()))
