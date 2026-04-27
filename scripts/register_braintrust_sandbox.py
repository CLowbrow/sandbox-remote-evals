#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT = "enron-email-agent"
DEFAULT_SANDBOX_NAME = "Enron Email Agent Sandbox"
DEFAULT_ENTRYPOINT = "./evals/enron_email_agent.eval.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the Enron Modal sandbox with Braintrust.")
    parser.add_argument("--project", default=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--name", default=os.environ.get("BRAINTRUST_SANDBOX_NAME", DEFAULT_SANDBOX_NAME))
    parser.add_argument("--snapshot-ref", default=os.environ.get("BRAINTRUST_SNAPSHOT_REF"))
    parser.add_argument("--entrypoint", default=os.environ.get("BRAINTRUST_ENTRYPOINT", DEFAULT_ENTRYPOINT))
    parser.add_argument(
        "--if-exists",
        choices=["error", "ignore", "replace"],
        default=os.environ.get("BRAINTRUST_IF_EXISTS", "replace"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.snapshot_ref:
        raise SystemExit("Pass --snapshot-ref im-... or set BRAINTRUST_SNAPSHOT_REF.")
    script = Path(__file__).with_suffix(".js")
    command = [
        "node",
        str(script),
        "--project",
        args.project,
        "--name",
        args.name,
        "--snapshot-ref",
        args.snapshot_ref,
        "--entrypoint",
        args.entrypoint,
        "--if-exists",
        args.if_exists,
    ]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
