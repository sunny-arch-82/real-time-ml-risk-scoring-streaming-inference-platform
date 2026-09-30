import os
import pandas as pd
import pytest
from riskflow.data import generate,load_events
from riskflow.features import offline_features,NUMERIC

@pytest.mark.spark
@pytest.mark.skipif(os.getenv("RISKFLOW_TEST_SPARK")!="1",reason="Opt-in Java/PySpark test")
def test_spark_matches_shared_semantics(tmp_path):
    pytest.importorskip("pyspark")
    from riskflow.spark import build_spark
    file=tmp_path/"input.csv";pd.DataFrame(list(generate(30))).to_csv(file,index=False)
    build_spark(file,tmp_path/"features")
    spark=pd.read_parquet(tmp_path/"features").sort_values("transaction_id").reset_index(drop=True)
    python=offline_features(load_events(file)).sort_values("transaction_id").reset_index(drop=True)
    for column in NUMERIC:
        assert spark[column].to_numpy()==pytest.approx(python[column].to_numpy(),abs=1e-7)
