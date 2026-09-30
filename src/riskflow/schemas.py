from datetime import datetime
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

class Contract(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class Transaction(Contract):
    transaction_id: str = Field(min_length=1,max_length=100)
    account_id: str = Field(min_length=1,max_length=100)
    timestamp: datetime
    amount: float = Field(gt=0,le=1_000_000)
    merchant_id: str = Field(min_length=1,max_length=100)
    category: Literal["groceries","electronics","travel","gaming","other"]
    country: Literal["US","GB","IN","DE","CA"]
    currency: Literal["USD","EUR","GBP","INR","CAD"]
    device_id: str = Field(min_length=1,max_length=100)
    trusted_device_id: str = Field(min_length=1,max_length=100)
    payment_method: Literal["card","wallet","bank"]
    @model_validator(mode="after")
    def aware_time(self):
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware")
        if self.timestamp.microsecond:
            raise ValueError("Event time precision must be whole seconds")
        return self
class LabeledTransaction(Transaction):
    fraud: Literal[0,1]
class Prediction(Contract):
    transaction_id: str
    probability: float = Field(ge=0,le=1)
    predicted_label: Literal[0,1]
    threshold: float = Field(ge=0,le=1)
    model_version: str
    feature_version: str
    features: dict[str,float|str]
    latency_ms: float = Field(ge=0)
    duplicate: bool = False
