# Demo Sandbox Env

This repo is for building Modal sandbox evals over the Enron dataset, comparing
embedding-backed retrieval against filesystem/ripgrep search.

## Current State

- `data/emails.csv` is the source CSV used for embeddings.
- `data/enron_mail.tar.gz` is the maildir tarball used for filesystem search.
- Local embedding generation is implemented in `scripts/ingest_enron_embeddings.py`.
- Postgres dump helper is implemented in `scripts/dump_postgres.sh`.
- The default embedding model is `BAAI/bge-small-en-v1.5`, producing normalized
  384-dimensional vectors.
- The maildir tarball was uploaded to a Modal v2 Volume named `enron-maildir` and
  extracted to `/maildir`.

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
