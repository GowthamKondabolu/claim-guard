from __future__ import annotations

import argparse
import json

from claimguard.config import load_settings
from claimguard.data.cms_synpuf import adapt_cms_synpuf_csv
from claimguard.pipeline import run_csv_training_pipeline, run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimGuard project commands")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train on generated synthetic claims")
    train_parser.add_argument(
        "--config", default="configs/base.yaml", help="YAML configuration path"
    )

    ingest_parser = commands.add_parser("ingest-cms", help="Convert a CMS DE-SynPUF CSV")
    ingest_parser.add_argument("--input", required=True, help="CMS source CSV")
    ingest_parser.add_argument("--output", required=True, help="Canonical output CSV")
    ingest_parser.add_argument(
        "--claim-type",
        required=True,
        choices=["inpatient", "outpatient", "carrier", "pharmacy", "pde"],
    )

    csv_parser = commands.add_parser("train-csv", help="Train on a canonical ClaimGuard CSV")
    csv_parser.add_argument("--input", required=True, help="Canonical ClaimGuard CSV")
    csv_parser.add_argument(
        "--config", default="configs/base.yaml", help="YAML configuration path"
    )
    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training_pipeline(load_settings(args.config))
        print(json.dumps(metrics, indent=2))
    elif args.command == "ingest-cms":
        claims = adapt_cms_synpuf_csv(args.input, args.claim_type, args.output)
        print(
            json.dumps(
                {
                    "claim_type": args.claim_type,
                    "canonical_claims": len(claims),
                    "output": args.output,
                },
                indent=2,
            )
        )
    elif args.command == "train-csv":
        metrics = run_csv_training_pipeline(args.input, load_settings(args.config))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
