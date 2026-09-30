import json
import os
import time
from pathlib import Path
import httpx
from .data import load_events
from .schemas import Transaction,Prediction

TOPIC="riskflow.transactions";OUTPUT="riskflow.predictions";DLQ="riskflow.dead-letter"
def create_topics():
    from confluent_kafka.admin import AdminClient,NewTopic
    admin=AdminClient({"bootstrap.servers":os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092")})
    for name,future in admin.create_topics([NewTopic(name,num_partitions=3,replication_factor=1) for name in (TOPIC,OUTPUT,DLQ)]).items():
        try:future.result()
        except Exception as exc:
            if "TOPIC_ALREADY_EXISTS" not in str(exc):raise

def produce(path,delay=.05):
    from confluent_kafka import Producer
    producer=Producer({"bootstrap.servers":os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092"),"enable.idempotence":True})
    errors=[]
    def delivery(error,message):
        if error:errors.append(str(error))
    for row in load_events(path,labeled=False).to_dict("records"):
        event=Transaction.model_validate({key:row[key] for key in Transaction.model_fields})
        producer.produce(TOPIC,key=event.account_id.encode(),value=event.model_dump_json().encode(),on_delivery=delivery)
        producer.poll(0);time.sleep(delay)
    remaining=producer.flush(30)
    if errors or remaining:raise RuntimeError(f"Kafka publish failures: {errors}; undelivered={remaining}")

def consume(group="riskflow-v1"):
    from confluent_kafka import Consumer,Producer
    broker=os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092")
    consumer=Consumer({"bootstrap.servers":broker,"group.id":group,"auto.offset.reset":"earliest","enable.auto.commit":False,"max.poll.interval.ms":300000})
    producer=Producer({"bootstrap.servers":broker,"enable.idempotence":True})
    consumer.subscribe([TOPIC])
    errors=[]
    def delivery(error,message):
        if error:errors.append(str(error))
    home=Path(os.getenv("RISKFLOW_HOME","artifacts"));home.mkdir(parents=True,exist_ok=True)
    api=os.getenv("RISKFLOW_API_URL","http://127.0.0.1:8000").rstrip("/")
    with httpx.Client(timeout=30) as client:
        try:
            while True:
                message=consumer.poll(1)
                if message is None:continue
                if message.error():raise RuntimeError(str(message.error()))
                try:
                    event=Transaction.model_validate_json(message.value())
                    response=client.post(api+"/predict",json=event.model_dump(mode="json"))
                    if response.status_code>=500:
                        # Stop without committing: model/API unavailable is retryable, not poison input.
                        raise RuntimeError(f"API unavailable: {response.status_code}; restart consumer when ready")
                    if response.status_code!=200:raise ValueError(f"API rejected event {response.status_code}: {response.text[:500]}")
                    prediction=Prediction.model_validate(response.json())
                    topic=OUTPUT;value=prediction.model_dump_json()
                except ValueError as exc:
                    topic=DLQ;value=json.dumps({"error":str(exc)[:1000],"raw":message.value().decode(errors="replace")[:10000],"partition":message.partition(),"offset":message.offset()})
                producer.produce(topic,key=message.key(),value=value.encode(),on_delivery=delivery)
                remaining=producer.flush(30)
                if remaining or errors:raise RuntimeError(f"Prediction publish failure: {errors}; undelivered={remaining}")
                consumer.commit(message=message,asynchronous=False)
                with (home/"stream-log.jsonl").open("a") as file:file.write(json.dumps({"topic":topic,"value":json.loads(value)})+"\n")
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close();producer.flush(5)
