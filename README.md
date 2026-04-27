# Demo Sandbox Env

This repo is for building Modal sandbox evals over the Enron dataset, comparing
embedding-backed retrieval against filesystem/ripgrep search.

## Current State

- `data/emails.csv` is the source CSV used for embeddings.
- `data/enron_mail.tar.gz` is the maildir tarball used for filesystem search.
- Local embedding generation is implemented in `scripts/ingest_enron_embeddings.py`.
- Postgres dump helper is implemented in `scripts/dump_postgres.sh`.
- Modal image and Braintrust sandbox build logic is in `modal_enron_sandbox.py`.
- The Braintrust Python eval is `evals/enron_email_agent.eval.py`.
- Braintrust sandbox registration helper is `scripts/register_braintrust_sandbox.py`.
- The default embedding model is `BAAI/bge-small-en-v1.5`, producing normalized
  384-dimensional vectors.
- The maildir tarball was uploaded to a Modal v2 Volume named `enron-maildir` and
  extracted to `/maildir`.
- The Postgres dump was uploaded to a Modal Volume named `enron-pg` as
  `/enron_embeddings.dump`.

## Local Setup

Use repo-local caches while working:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export HF_HOME="$PWD/.hf-cache"
export DATABASE_URL="postgresql:///enron_embeddings"
```

Install dependencies:

```bash
uv sync
```

The project currently depends on Modal, Braintrust, OpenAI, OpenAI Agents SDK,
psycopg, pgvector, sentence-transformers, and torch.

For the local test database, make sure the `alex` Postgres role and target DB exist:

```bash
sudo -u postgres psql -c "do \$\$ begin create role alex login createdb; exception when duplicate_object then alter role alex createdb; end \$\$;"
createdb enron_embeddings
```

Enable pgvector once as the Postgres superuser:

```bash
sudo -u postgres psql -d enron_embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Check CUDA from WSL:

```bash
nvidia-smi
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Embeddings Into Postgres

Smoke test:

```bash
uv run python scripts/ingest_enron_embeddings.py \
  --database-url "$DATABASE_URL" \
  --device cuda \
  --drop-existing \
  --limit 1000
```

Full load:

```bash
uv run python scripts/ingest_enron_embeddings.py \
  --database-url "$DATABASE_URL" \
  --device cuda \
  --drop-existing \
  --batch-size 128 \
  --embed-batch-size 256
```

With the standard Enron CSV, expect about 517k emails and roughly 650k-1M chunks,
depending on chunking. With `--batch-size 128`, the full load is about 4,043 batches.

Health checks:

```bash
psql "$DATABASE_URL" -c "
select
  (select count(*) from enron_emails) as emails,
  (select count(*) from enron_email_chunks) as chunks,
  round((select count(*) from enron_email_chunks)::numeric / nullif((select count(*) from enron_emails), 0), 3) as chunks_per_email;
"

psql "$DATABASE_URL" -c "
select
  count(*) as chunks_checked,
  min(vector_dims(embedding)) as min_dims,
  max(vector_dims(embedding)) as max_dims,
  min(length(content)) as min_content_len,
  percentile_cont(0.5) within group (order by length(content)) as median_content_len,
  max(length(content)) as max_content_len
from enron_email_chunks;
"
```

Dump the DB:

```bash
./scripts/dump_postgres.sh dumps/enron_embeddings.dump
```

Upload the dump to Modal:

```bash
modal volume create enron-pg
modal volume put enron-pg dumps/enron_embeddings.dump /enron_embeddings.dump
modal volume ls enron-pg /
```

## Postgres Schema

`enron_emails` has one row per email:

```sql
id bigserial primary key,
source_file text unique not null,
message_id text,
sent_at timestamptz,
sender text,
recipients text[] not null default '{}',
subject text,
body_sha256 text not null,
raw_len integer not null
```

`enron_email_chunks` has one row per embedded chunk:

```sql
id bigserial primary key,
email_id bigint references enron_emails(id) on delete cascade,
chunk_index integer not null,
content text not null,
embedding vector(384) not null,
unique (email_id, chunk_index)
```

The embedded text includes `Subject`, `From`, first 20 `To` recipients, and the
body chunk. Exact metadata filters should still use SQL columns.

## Maildir Files Into Modal

The tarball already contains a top-level `maildir/` directory:

```bash
tar -tf data/enron_mail.tar.gz | head
```

Create and populate the v2 Volume:

```bash
modal volume create enron-maildir --version=2
modal volume put enron-maildir data/enron_mail.tar.gz /enron_mail.tar.gz
modal volume ls enron-maildir /
```

Extract in a Modal shell:

```bash
modal shell --volume enron-maildir
cd /mnt/enron-maildir
tar -xzf enron_mail.tar.gz
find maildir -type f | wc -l
sync
exit
```

Verify from local WSL:

```bash
modal volume ls enron-maildir /
modal volume ls enron-maildir /maildir
```

Mount in Modal for ripgrep:

```python
maildir_vol = modal.Volume.from_name("enron-maildir")

@app.function(volumes={"/mnt/enron-maildir": maildir_vol.read_only()})
def run_eval():
    # Search root: /mnt/enron-maildir/maildir
    ...
```

## Modal Postgres Image For Evals

`modal_enron_sandbox.py` defines a Modal image with:

- Python 3.12
- Node.js and npm
- PostgreSQL 18 from PGDG apt packages
- `postgresql-18-pgvector`
- Braintrust, OpenAI, OpenAI Agents SDK, psycopg, pgvector, sentence-transformers, torch
- cached `BAAI/bge-small-en-v1.5` embedding model files
- `evals/enron_email_agent.eval.py` copied to `/app/evals/enron_email_agent.eval.py`
- the restored `enron_embeddings` database baked into the image filesystem

The image build step mounts the existing `enron-pg` volume read-only at
`/mnt/enron-pg`, restores `/mnt/enron-pg/enron_embeddings.dump` into a fresh
Postgres 18 cluster, runs `ANALYZE` and `CHECKPOINT`, then stops Postgres. Later
eval containers start from that image, so startup only needs to launch Postgres
against the already-restored data directory.

Build and verify:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run modal run modal_enron_sandbox.py
```

Build and print the Modal image ID for Braintrust:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run modal run modal_enron_sandbox.py::print_image_id
```

This prints a final `im-...` image ID. Use the final printed image ID, not any
intermediate image IDs shown in build logs.

The healthcheck starts Postgres, checks row counts, verifies the pgvector
extension and vector index, confirms the maildir volume is mounted, prints the
Node version, and stops Postgres.

Inside eval code, use both `postgres_image` and `eval_volumes` so Postgres starts
from the restored image filesystem while the original maildir files are mounted
read-only:

```python
import subprocess

from modal_enron_sandbox import app, eval_volumes, postgres_image


@app.function(image=postgres_image, volumes=eval_volumes)
def run_eval():
    subprocess.run(["start-enron-postgres"], check=True)

    # Vector DB:
    # postgresql:///enron_embeddings?host=/var/run/postgresql&user=postgres

    # Maildir files:
    # /mnt/enron-maildir/maildir
```

The maildir volume is still mounted separately and should be searched at:

```text
/mnt/enron-maildir/maildir
```

Large local `data/` and `dumps/` files are excluded from Modal build context by
`.dockerignore`; the dump should come from the Modal volume, not local upload.

## Braintrust Sandbox And Eval

The current Braintrust eval uses the Python OpenAI Agents SDK. It defines one
tool, `search_email_chunks`, which:

- starts bundled Postgres when running inside the Modal image
- embeds the dataset input question with `BAAI/bge-small-en-v1.5`
- queries `enron_email_chunks` with pgvector cosine distance
- returns source file, sender, recipients, subject, chunk index, excerpt, and distance

The eval parameters are intentionally minimal:

- `model`: OpenAI model for the agent, default `gpt-5-mini`

The question must come from the Braintrust dataset row `input`. There is no
`query` parameter anymore.

The eval includes:

- `setup_openai_agents(project_name="enron-email-agent")` for Braintrust tracing
- Python logging around startup, DB URL selection, search query, result count, and tool errors
- Braintrust span metadata for `model`, `query`, `database_url`, successful search details, and search errors

Local smoke checks:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run python -m py_compile \
  modal_enron_sandbox.py \
  evals/enron_email_agent.eval.py \
  scripts/register_braintrust_sandbox.py

UV_CACHE_DIR="$PWD/.uv-cache" uv run braintrust eval \
  evals/enron_email_agent.eval.py \
  --list \
  --no-send-logs
```

A full local run against WSL Postgres needs network access for OpenAI and
Hugging Face if the embedding model is not already cached:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" \
OPENAI_AGENTS_DISABLE_TRACING=1 \
uv run braintrust eval evals/enron_email_agent.eval.py \
  --no-send-logs \
  --no-progress-bars \
  --terminate-on-failure
```

Register or update the Braintrust sandbox after building a new image:

```bash
BRAINTRUST_SNAPSHOT_REF="im-your-new-image-id" \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run python scripts/register_braintrust_sandbox.py
```

By default this registers:

- project: `enron-email-agent`
- sandbox name: `Enron Email Agent Sandbox`
- entrypoint: `./evals/enron_email_agent.eval.py`
- `if_exists`: `replace`

Override the project or sandbox name if needed:

```bash
BRAINTRUST_SNAPSHOT_REF="im-your-new-image-id" \
UV_CACHE_DIR="$PWD/.uv-cache" \
uv run python scripts/register_braintrust_sandbox.py \
  --project "your-project" \
  --name "your-sandbox-name"
```

If Braintrust UI behavior looks stale after re-registering, create a new sandbox
name and select that exact name in the UI. We saw the UI cache an older sandbox
selection once, which made it look like a rebuilt image was still using old code.

## Current Runtime Notes

- In Modal, Postgres is started by `/usr/local/bin/start-enron-postgres`.
- Runtime Postgres connection defaults to
  `postgresql:///enron_embeddings?host=/var/run/postgresql`.
- Local WSL testing defaults to `postgresql:///enron_embeddings` and assumes
  local Postgres is already running.
- The maildir volume is mounted by `eval_volumes` at `/mnt/enron-maildir`; files
  live under `/mnt/enron-maildir/maildir`.
- The current Braintrust eval only uses pgvector search. Filesystem/ripgrep
  comparison still needs to be added to the eval.
