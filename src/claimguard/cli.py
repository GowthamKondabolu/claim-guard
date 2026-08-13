from __future__ import annotations

import argparse
import json

from claimguard.config import load_settings
from claimguard.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimGuard project commands")
    parser.add_argument("command", choices=["train"], help="Command to run")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML configuration path")
    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training_pipeline(load_settings(args.config))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

