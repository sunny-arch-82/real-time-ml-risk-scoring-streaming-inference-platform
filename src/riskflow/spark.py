"""Real Spark SQL windows; local mode optional, same prior-event semantics."""
from .features import FEATURE_VERSION,RISK
from .data import load_events

def build_spark(input_path,output_path,master="local[2]"):
    from pyspark.sql import SparkSession,Window,functions as F
    spark=SparkSession.builder.master(master).appName("RiskFlowOfflineFeatures").config("spark.sql.session.timeZone","UTC").getOrCreate()
    try:
        # Local demo validation; remote/large datasets need equivalent distributed validation.
        frame=load_events(input_path)
        records=frame.to_dict("records")
        for row in records:
            row["timestamp"]=row["timestamp"].to_pydatetime()
        df=spark.createDataFrame(records).withColumn("event_second",F.unix_timestamp("timestamp"))
        ordering=Window.partitionBy("account_id").orderBy("event_second")
        day=ordering.rangeBetween(-86400,-1);hour=ordering.rangeBetween(-3600,-1);five=ordering.rangeBetween(-300,-1)
        df=df.withColumn("count_5m",F.count("transaction_id").over(five)).withColumn("count_1h",F.count("transaction_id").over(hour)).withColumn("count_24h",F.count("transaction_id").over(day))
        df=df.withColumn("spend_1h",F.coalesce(F.sum("amount").over(hour),F.lit(0.0))).withColumn("mean_24h",F.coalesce(F.avg("amount").over(day),F.lit(0.0))).withColumn("std_24h",F.coalesce(F.stddev_pop("amount").over(day),F.lit(0.0)))
        df=df.withColumn("unique_merchants_1h",F.size(F.collect_set("merchant_id").over(hour)))
        df=df.withColumn("previous_second",F.lag("event_second").over(ordering)).withColumn("previous_device",F.lag("device_id").over(ordering)).withColumn("previous_country",F.lag("country").over(ordering))
        active=(F.col("event_second")-F.col("previous_second")<=86400)&F.col("previous_second").isNotNull()
        df=df.withColumn("time_since_previous",F.when(active,F.col("event_second")-F.col("previous_second")).otherwise(86400)).withColumn("device_changed",F.when(active&(F.col("device_id")!=F.col("previous_device")),1).otherwise(0)).withColumn("country_changed",F.when(active&(F.col("country")!=F.col("previous_country")),1).otherwise(0))
        risk_map=F.create_map(*[value for key,score in RISK.items() for value in (F.lit(key),F.lit(score))])
        df=df.withColumn("merchant_risk",risk_map[F.col("category")]).withColumn("device_mismatch",(F.col("device_id")!=F.col("trusted_device_id")).cast("int")).withColumn("hour",F.hour("timestamp"))
        df=df.withColumn("amount_zscore",F.when(F.col("std_24h")>1e-9,(F.col("amount")-F.col("mean_24h"))/F.col("std_24h")).otherwise(0.0)).withColumn("feature_version",F.lit(FEATURE_VERSION))
        df=df.drop("event_second","previous_second","previous_device","previous_country")
        df.write.mode("errorifexists").parquet(str(output_path))
    finally:
        spark.stop()
