import { spawnSync } from "node:child_process";
import { userInfo } from "node:os";

import { OpenAIAgentsTraceProcessor } from "@braintrust/openai-agents";
import { Agent, addTraceProcessor, run, tool } from "@openai/agents";
import { Eval, initLogger } from "braintrust";
import pg from "pg";
import { z } from "zod";

const { Client } = pg;

const DATABASE_NAME = "enron_embeddings";
const DATABASE_URL = process.env.DATABASE_URL || "postgresql:///enron_embeddings";
const EMBEDDING_MODEL = process.env.ENRON_EMBEDDING_MODEL || "BAAI/bge-small-en-v1.5";
const PROJECT_NAME = process.env.BRAINTRUST_PROJECT || "enron-email-agent";
const MODAL_DATABASE_URL = "postgresql:///enron_embeddings?host=/var/run/postgresql";

initLogger({
  projectName: PROJECT_NAME,
  apiKey: process.env.BRAINTRUST_API_KEY,
});
addTraceProcessor(new OpenAIAgentsTraceProcessor());

let postgresStarted = false;

function which(command) {
  return spawnSync("sh", ["-lc", `command -v ${command}`], {
    stdio: "ignore",
  }).status === 0;
}

function tailText(value, limit) {
  return String(value || "").trim().slice(-limit);
}

function runtimeDatabaseUser() {
  return process.env.ENRON_DATABASE_USER || userInfo().username;
}

function startPostgres(span) {
  if (postgresStarted) {
    return;
  }
  if (!which("start-enron-postgres")) {
    console.info("No start-enron-postgres helper found; assuming Postgres is already running.");
    postgresStarted = true;
    return;
  }

  console.info("Starting bundled Postgres cluster with start-enron-postgres.");
  const result = spawnSync("start-enron-postgres", [], {
    encoding: "utf8",
    timeout: 45_000,
  });
  if (result.status !== 0) {
    const diagnostics = {
      command: "start-enron-postgres",
      status: result.status,
      signal: result.signal,
      stdout: tailText(result.stdout, 4000),
      stderr: tailText(result.stderr, 4000),
    };
    span?.log({ metadata: { postgres_start_error: diagnostics } });
    throw new Error(`start-enron-postgres failed: ${JSON.stringify(diagnostics)}`);
  }
  postgresStarted = true;
}

function databaseUrl() {
  if (which("start-enron-postgres")) {
    return `${MODAL_DATABASE_URL}&user=${encodeURIComponent(runtimeDatabaseUser())}`;
  }
  return DATABASE_URL;
}

function pgConfig() {
  if (which("start-enron-postgres")) {
    return {
      database: DATABASE_NAME,
      host: "/var/run/postgresql",
      user: runtimeDatabaseUser(),
    };
  }

  if (DATABASE_URL.startsWith("postgresql:///")) {
    const database = DATABASE_URL.slice("postgresql:///".length).split("?")[0] || DATABASE_NAME;
    return { database };
  }

  return { connectionString: DATABASE_URL };
}

function vectorLiteral(values) {
  return `[${values.map((value) => Number(value).toFixed(8)).join(",")}]`;
}

function embedQuery(query) {
  const script = `
import json
import os
import sys
from functools import lru_cache
from sentence_transformers import SentenceTransformer

model_name = os.environ.get("ENRON_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
payload = json.loads(sys.stdin.read())
model = SentenceTransformer(model_name, device="cpu")
embedding = model.encode([payload["query"]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
print(json.dumps([float(value) for value in embedding]))
`;
  const result = spawnSync("python", ["-c", script], {
    input: JSON.stringify({ query }),
    encoding: "utf8",
    maxBuffer: 1024 * 1024 * 8,
    env: {
      ...process.env,
      ENRON_EMBEDDING_MODEL: EMBEDDING_MODEL,
    },
  });
  if (result.status !== 0) {
    throw new Error(`Embedding query failed: ${tailText(result.stderr, 4000)}`);
  }
  return JSON.parse(result.stdout);
}

async function searchEmailChunks({ query, limit }, rootSpan) {
  const dbUrl = databaseUrl();
  try {
    startPostgres(rootSpan);
    const safeLimit = Math.max(1, Math.min(Number.parseInt(limit ?? 10, 10) || 10, 20));
    console.info(`Searching Enron chunks. query=${JSON.stringify(query)} limit=${safeLimit} database_url=${dbUrl}`);

    const vector = vectorLiteral(embedQuery(query));
    const client = new Client(pgConfig());
    await client.connect();
    try {
      const { rows } = await client.query(
        `
          select
            e.source_file,
            e.sent_at,
            e.sender,
            e.recipients,
            e.subject,
            c.chunk_index,
            left(c.content, 1600) as excerpt,
            c.embedding <=> $1::vector as distance
          from enron_email_chunks c
          join enron_emails e on e.id = c.email_id
          order by c.embedding <=> $1::vector
          limit $2
        `,
        [vector, safeLimit],
      );

      const results = rows.map((row) => ({
        source_file: row.source_file,
        sent_at: row.sent_at instanceof Date ? row.sent_at.toISOString() : row.sent_at,
        sender: row.sender,
        recipients: (row.recipients || []).slice(0, 12),
        subject: row.subject,
        chunk_index: row.chunk_index,
        excerpt: row.excerpt,
        distance: Number(row.distance),
      }));
      rootSpan.log({
        metadata: {
          search_email_chunks: {
            query,
            limit: safeLimit,
            result_count: results.length,
            database_url: dbUrl,
          },
        },
      });
      console.info(`Enron search returned ${results.length} chunks.`);
      return JSON.stringify(results);
    } finally {
      await client.end();
    }
  } catch (error) {
    rootSpan.log({
      metadata: {
        search_email_chunks_error: {
          query,
          database_url: dbUrl,
          error_type: error?.constructor?.name || "Error",
          error: error?.message || String(error),
        },
      },
    });
    throw error;
  }
}

function parameterValue(parameters, name, defaultValue) {
  const value = parameters?.[name] ?? defaultValue;
  if (value && typeof value === "object" && "value" in value) {
    return value.value;
  }
  return value;
}

async function task(input, hooks) {
  const query = String(input || "");
  if (!query.trim()) {
    throw new Error("Dataset input is required and should contain the question to ask about the emails.");
  }

  const model = String(parameterValue(hooks.parameters, "model", "gpt-5-mini"));
  const dbUrl = databaseUrl();
  const metadata = hooks.metadata || {};
  metadata.model = model;
  metadata.query = query;
  metadata.database_url = dbUrl;
  hooks.metadata = metadata;
  hooks.span.log({ metadata: { model, query, database_url: dbUrl } });

  const searchTool = tool({
    name: "search_email_chunks",
    description: "Search Enron email chunks by semantic similarity and return source metadata plus excerpts.",
    parameters: z.object({
      query: z.string(),
      limit: z.number().int().min(1).max(20).default(10),
    }),
    async execute(args) {
      return searchEmailChunks(args, hooks.span);
    },
  });

  const agent = new Agent({
    name: "Enron email analyst",
    model,
    instructions: [
      "Answer questions about the Enron email corpus. Use search_email_chunks before answering.",
      "The search tool is semantic vector search over chunks built from Subject, From, the first To recipients, and body text.",
      "The search string is embedded as one vector, not parsed as a query language. Extra filler words can dilute the important concept.",
      "Use compact passage-like phrases or exact likely words that would appear in matching emails. For lexical concepts, prefer sharp queries like 'joke' or 'jokes' over broad paraphrases like 'emails that include jokes or humor, people telling jokes or forwarding jokes'.",
      "Do not use Boolean, regex, SQL, OR/AND lists, quoted synonym lists, wildcard syntax, or long keyword chains.",
      "Prefer one concise query, for example: 'joke' for joke-finding tasks, or 'California energy prices joke' when the question needs both topic and tone.",
      "If the first search is weak, make at most one follow-up search with a meaningfully different natural-language phrasing or a narrower entity/time/topic. Do not issue many near-duplicate searches.",
      "Use metadata in the returned results, including source_file, sender, recipients, subject, sent_at, and chunk_index, to reason about provenance. Do not invent metadata filters that the tool does not support.",
      "Ground the answer in returned email excerpts and include source_file values for important claims.",
      "If the retrieved evidence is weak or absent, say that directly.",
    ].join(" "),
    tools: [searchTool],
  });

  return hooks.span.traced(
    async () => {
      const result = await run(agent, `Question: ${query}`, { maxTurns: 6 });
      return String(result.finalOutput ?? "");
    },
    { name: "openai_agents_runner" },
  );
}

Eval(PROJECT_NAME, {
  data: async () => [{ input: "" }],
  task,
  scores: [],
  parameters: {
    model: z.string().default("gpt-5-mini").describe("OpenAI model for the email-answering agent."),
  },
});
