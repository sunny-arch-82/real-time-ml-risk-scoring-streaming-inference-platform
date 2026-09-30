import argparse
import json
import os
from pathlib import Path
import pandas as pd
from .config import load_config,TrainConfig,Policy
from .data import write_generated,load_events
from .features import offline_features
from .training import train
from .registry import Registry
from .batch import batch_score,export_monitoring
from .serving import StateStore
from .drift import analyze_drift,retraining_decision

def main(argv=None):
    parser=argparse.ArgumentParser(description="RiskFlow synthetic ML lifecycle")
    commands=parser.add_subparsers(dest="command",required=True)
    g=commands.add_parser("generate");g.add_argument("--output",required=True);g.add_argument("--rows",type=int,default=3000);g.add_argument("--seed",type=int,default=20260929);g.add_argument("--start",default="2026-01-01T00:00:00+00:00");g.add_argument("--drift",action="store_true");g.add_argument("--concept-drift",action="store_true");g.add_argument("--prefix",default="hist")
    f=commands.add_parser("features");f.add_argument("--input",required=True);f.add_argument("--output",required=True);f.add_argument("--engine",choices=["python","spark"],default="python")
    for name in ("train","tune"):
        t=commands.add_parser(name);t.add_argument("--data",default="data/sample/transactions.csv");t.add_argument("--config",default="configs/train.yaml");t.add_argument("--features",help="Precomputed versioned Python/Spark Parquet features")
    p=commands.add_parser("promote");p.add_argument("--version",required=True);p.add_argument("--policy",default="configs/policy.yaml")
    r=commands.add_parser("rollback");r.add_argument("--version",required=True)
    commands.add_parser("model-info")
    b=commands.add_parser("batch");b.add_argument("--input",required=True);b.add_argument("--output",required=True);b.add_argument("--version")
    s=commands.add_parser("warmup");s.add_argument("--data",default="data/sample/transactions.csv")
    e=commands.add_parser("export-monitoring");e.add_argument("--output",default="artifacts/monitoring.csv")
    d=commands.add_parser("drift");d.add_argument("--recent",required=True);d.add_argument("--policy",default="configs/policy.yaml");d.add_argument("--output",default="artifacts/drift.json");d.add_argument("--new-labels",type=int,default=0);d.add_argument("--scheduled",action="store_true")
    commands.add_parser("kafka-topics")
    prod=commands.add_parser("produce");prod.add_argument("--data",default="data/sample/live_transactions.csv");prod.add_argument("--delay",type=float,default=.05)
    cons=commands.add_parser("consume");cons.add_argument("--group",default="riskflow-v1")
    args=parser.parse_args(argv)
    try:
        if args.command=="generate":
            if args.rows<1:raise ValueError("rows must be positive")
            write_generated(args.output,args.rows,seed=args.seed,start=args.start,drift=args.drift,concept_drift=args.concept_drift,prefix=args.prefix);print(args.output)
        elif args.command=="features":
            if args.engine=="spark":
                from .spark import build_spark
                build_spark(args.input,args.output)
            else:
                output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
                offline_features(load_events(args.input)).to_parquet(output,index=False)
            print(args.output)
        elif args.command in {"train","tune"}:print(json.dumps(train(args.data,load_config(args.config,TrainConfig),tune=args.command=="tune",feature_path=args.features),indent=2))
        elif args.command=="promote":
            result=Registry().promote(args.version,load_config(args.policy,Policy));print(json.dumps(result,indent=2));return 0 if result["promoted"] else 1
        elif args.command=="rollback":
            result=Registry().promote(args.version,Policy(),rollback=True);print(json.dumps(result,indent=2));return 0 if result["promoted"] else 1
        elif args.command=="model-info":print(json.dumps(Registry().current(),indent=2))
        elif args.command=="batch":
            result=batch_score(args.input,args.output,args.version);print(f"Scored {len(result)} rows: {args.output}")
        elif args.command=="warmup":
            home=Path(os.getenv("RISKFLOW_HOME","artifacts"));StateStore(home/"online-state.db").seed(load_events(args.data,labeled=False));print("History persisted; no model training or scoring")
        elif args.command=="export-monitoring":
            frame=export_monitoring(Path(os.getenv("RISKFLOW_HOME","artifacts"))/"online-state.db",args.output);print(f"Exported {len(frame)} successful predictions")
        elif args.command=="drift":
            _,metadata=Registry().load();recent=pd.read_parquet(args.recent) if args.recent.endswith(".parquet") else pd.read_csv(args.recent)
            policy=load_config(args.policy,Policy);report=analyze_drift(metadata["reference"],recent,policy,metadata["threshold"])
            report["retraining"]=retraining_decision(report,policy,args.new_labels,args.scheduled)
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report["retraining"],indent=2))
        elif args.command=="kafka-topics":
            from .streaming import create_topics
            create_topics();print("Topics ready")
        elif args.command=="produce":
            from .streaming import produce
            produce(args.data,args.delay)
        elif args.command=="consume":
            from .streaming import consume
            consume(args.group)
        return 0
    except (ValueError,OSError,ImportError) as exc:
        print(f"riskflow: {exc}");return 2

def entrypoint():raise SystemExit(main())
if __name__=="__main__":entrypoint()
