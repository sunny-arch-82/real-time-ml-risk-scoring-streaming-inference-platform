import uuid
import pytest
from fastapi.testclient import TestClient
from riskflow.registry import Registry
from riskflow.features import FEATURE_VERSION
from riskflow.api import app

def test_bundle_serialization_and_checksum(tmp_path):
    version=uuid.uuid4().hex;registry=Registry(tmp_path)
    registry.save(version,{"threshold":.2},{"version":version,"feature_version":FEATURE_VERSION})
    assert registry.load(version)[0]["threshold"]==.2
    (registry.directory(version)/"bundle.joblib").write_bytes(b"bad")
    with pytest.raises(ValueError,match="checksum"):registry.load(version)
def test_api_unready_and_schema(monkeypatch,tmp_path):
    monkeypatch.setenv("RISKFLOW_HOME",str(tmp_path/"artifacts"));monkeypatch.setenv("RISKFLOW_REGISTRY",str(tmp_path/"models"))
    monkeypatch.setattr("riskflow.api.service",None)
    client=TestClient(app)
    assert client.get("/health").status_code==200
    assert client.get("/ready").status_code==503
    assert client.post("/predict",json={"amount":-1}).status_code==422
