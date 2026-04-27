#!/usr/bin/env node
import { registerSandbox } from "braintrust";

const DEFAULT_PROJECT = "enron-email-agent";
const DEFAULT_SANDBOX_NAME = "Enron Email Agent Sandbox";
const DEFAULT_ENTRYPOINT = "./evals/enron_email_agent.eval.js";

function parseArgs(argv) {
  const args = {
    project: process.env.BRAINTRUST_PROJECT || DEFAULT_PROJECT,
    name: process.env.BRAINTRUST_SANDBOX_NAME || DEFAULT_SANDBOX_NAME,
    snapshotRef: process.env.BRAINTRUST_SNAPSHOT_REF,
    entrypoint: process.env.BRAINTRUST_ENTRYPOINT || DEFAULT_ENTRYPOINT,
    ifExists: process.env.BRAINTRUST_IF_EXISTS || "replace",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const readValue = () => {
      index += 1;
      if (index >= argv.length) {
        throw new Error(`Missing value for ${arg}`);
      }
      return argv[index];
    };

    if (arg === "--help" || arg === "-h") {
      args.help = true;
    } else if (arg === "--project") {
      args.project = readValue();
    } else if (arg === "--name") {
      args.name = readValue();
    } else if (arg === "--snapshot-ref") {
      args.snapshotRef = readValue();
    } else if (arg === "--entrypoint") {
      args.entrypoint = readValue();
    } else if (arg === "--if-exists") {
      args.ifExists = readValue();
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function printHelp() {
  console.log(`Register the Enron Modal sandbox with Braintrust.

Options:
  --project NAME        Braintrust project. Default: ${DEFAULT_PROJECT}
  --name NAME           Sandbox group name. Default: ${DEFAULT_SANDBOX_NAME}
  --snapshot-ref REF    Modal image/snapshot ref, e.g. im-...
  --entrypoint PATH     Eval entrypoint. Default: ${DEFAULT_ENTRYPOINT}
  --if-exists MODE      error, ignore, or replace. Default: replace
`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    return;
  }
  if (!args.snapshotRef) {
    throw new Error("Pass --snapshot-ref im-... or set BRAINTRUST_SNAPSHOT_REF.");
  }
  if (!["error", "ignore", "replace"].includes(args.ifExists)) {
    throw new Error("--if-exists must be one of: error, ignore, replace");
  }

  const result = await registerSandbox({
    name: args.name,
    project: args.project,
    sandbox: {
      provider: "modal",
      snapshotRef: args.snapshotRef,
    },
    entrypoints: [args.entrypoint],
    ifExists: args.ifExists,
  });
  console.log(JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
