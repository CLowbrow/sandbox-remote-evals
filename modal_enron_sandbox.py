from __future__ import annotations

import os
import subprocess
import modal


APP_NAME = "enron-sandbox"
POSTGRES_MAJOR = "18"
DATABASE_NAME = "enron_embeddings"
DUMP_PATH = "/mnt/enron-pg/enron_embeddings.dump"
MAILDIR_ROOT = "/mnt/enron-maildir/maildir"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

app = modal.App(APP_NAME)

enron_pg_volume = modal.Volume.from_name("enron-pg")
enron_maildir_volume = modal.Volume.from_name("enron-maildir")

eval_volumes = {
    "/mnt/enron-maildir": enron_maildir_volume.read_only(),
}


def _run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def _run_shell(command: str, **kwargs) -> None:
    subprocess.run(["bash", "-lc", command], check=True, **kwargs)


def _install_runtime_helpers() -> None:
    start_script = f"""\
#!/usr/bin/env bash
set -euo pipefail

if pg_isready -q -h /var/run/postgresql -d {DATABASE_NAME}; then
  exit 0
fi

pg_ctlcluster {POSTGRES_MAJOR} main start

for _ in $(seq 1 60); do
  if pg_isready -q -h /var/run/postgresql -d {DATABASE_NAME}; then
    exit 0
  fi
  sleep 0.5
done

pg_ctlcluster {POSTGRES_MAJOR} main status || true
exit 1
"""
    stop_script = f"""\
#!/usr/bin/env bash
set -euo pipefail

pg_ctlcluster {POSTGRES_MAJOR} main stop --mode=fast
"""
    with open("/usr/local/bin/start-enron-postgres", "w", encoding="utf-8") as handle:
        handle.write(start_script)
    with open("/usr/local/bin/stop-enron-postgres", "w", encoding="utf-8") as handle:
        handle.write(stop_script)
    os.chmod("/usr/local/bin/start-enron-postgres", 0o755)
    os.chmod("/usr/local/bin/stop-enron-postgres", 0o755)


def restore_postgres_dump() -> None:
    """Image build step: restore the Enron pgvector dump into the image filesystem."""
    if not os.path.exists(DUMP_PATH):
        raise FileNotFoundError(f"Expected dump at {DUMP_PATH}")

    _install_runtime_helpers()

    _run_shell(f"pg_dropcluster --stop {POSTGRES_MAJOR} main || true")
    _run(["pg_createcluster", POSTGRES_MAJOR, "main", "--start-conf=manual"])
    _run(["pg_ctlcluster", POSTGRES_MAJOR, "main", "start"])

    try:
        _run(["sudo", "-u", "postgres", "createdb", DATABASE_NAME])
        _run(
            [
                "sudo",
                "-u",
                "postgres",
                "psql",
                "--dbname",
                DATABASE_NAME,
                "--command",
                "CREATE EXTENSION IF NOT EXISTS vector;",
            ]
        )
        _run(
            [
                "sudo",
                "-u",
                "postgres",
                "pg_restore",
                "--dbname",
                DATABASE_NAME,
                "--no-owner",
                "--no-privileges",
                "--exit-on-error",
                "--jobs",
                str(os.cpu_count() or 4),
                DUMP_PATH,
            ]
        )
        _run(
            [
                "sudo",
                "-u",
                "postgres",
                "psql",
                "--dbname",
                DATABASE_NAME,
                "--command",
                "ANALYZE;",
            ]
        )
        _run(
            [
                "sudo",
                "-u",
                "postgres",
                "psql",
                "--dbname",
                DATABASE_NAME,
                "--command",
                "CHECKPOINT;",
            ]
        )
    finally:
        _run_shell(f"pg_ctlcluster {POSTGRES_MAJOR} main stop --mode=fast || true")
        _run(["sync"])


def download_embedding_model() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    model.encode(["dimension probe"], convert_to_numpy=True, normalize_embeddings=True)


postgres_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ca-certificates", "curl", "gnupg", "sudo")
    .run_commands(
        (
            "install -d /usr/share/postgresql-common/pgdg && "
            "curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc "
            "-o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc && "
            ". /etc/os-release && "
            'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] '
            'https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" '
            "> /etc/apt/sources.list.d/pgdg.list && "
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
            "nodejs npm postgresql-18 postgresql-18-pgvector postgresql-client-18 && "
            "rm -rf /var/lib/apt/lists/*"
        )
    )
    .pip_install(
        "braintrust==0.16.0",
        "openai==2.32.0",
        "openai-agents==0.14.6",
        "pgvector>=0.3.6",
        "psycopg[binary]>=3.2.6",
        "sentence-transformers>=3.4.1",
        "torch>=2.7.0",
    )
    .add_local_file(
        "evals/enron_email_agent.eval.py",
        "/app/evals/enron_email_agent.eval.py",
        copy=True,
    )
    .workdir("/app")
    .run_function(download_embedding_model, timeout=60 * 30)
    .run_function(
        restore_postgres_dump,
        volumes={"/mnt/enron-pg": enron_pg_volume.read_only()},
        timeout=60 * 60 * 3,
    )
    .env(
        {
            "DATABASE_URL": f"postgresql:///{DATABASE_NAME}?host=/var/run/postgresql",
            "PGHOST": "/var/run/postgresql",
            "PGDATABASE": DATABASE_NAME,
            "PGUSER": "postgres",
            "MAILDIR_ROOT": MAILDIR_ROOT,
            "ENRON_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "HF_HOME": "/root/.cache/huggingface",
        }
    )
)


@app.function(
    image=postgres_image,
    volumes=eval_volumes,
    timeout=300,
)
def postgres_healthcheck() -> dict[str, int | str]:
    _run(["start-enron-postgres"])
    try:
        query = """
        select
          (select count(*) from enron_emails) as emails,
          (select count(*) from enron_email_chunks) as chunks,
          (select count(*) from pg_extension where extname = 'vector') as vector_extension,
          (select count(*) from pg_indexes where indexname = 'enron_email_chunks_embedding_ivfflat_idx') as vector_indexes
        """
        output = subprocess.check_output(
            [
                "psql",
                "--tuples-only",
                "--no-align",
                "--field-separator=,",
                "--command",
                query,
            ],
            text=True,
        ).strip()
        emails, chunks, vector_extension, vector_indexes = [int(value) for value in output.split(",")]
        maildir_check = subprocess.check_output(
            ["bash", "-lc", f"test -d {MAILDIR_ROOT} && find {MAILDIR_ROOT} -type f | head -n 1"],
            text=True,
        ).strip()
        node_version = subprocess.check_output(["node", "--version"], text=True).strip()
        return {
            "emails": emails,
            "chunks": chunks,
            "vector_extension": vector_extension,
            "vector_indexes": vector_indexes,
            "maildir_sample": maildir_check,
            "node_version": node_version,
        }
    finally:
        _run_shell("stop-enron-postgres || true")


@app.local_entrypoint()
def main() -> None:
    print(postgres_healthcheck.remote())


@app.local_entrypoint()
def print_image_id() -> None:
    built_image = postgres_image.build(app)
    print(built_image.object_id)
