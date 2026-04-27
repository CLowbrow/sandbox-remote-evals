from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from contextvars import ContextVar
from functools import lru_cache
from getpass import getuser
from typing import Any
from urllib.parse import quote

import psycopg
from agents import Agent, Runner, function_tool
from braintrust import Eval
from braintrust.integrations.openai_agents import setup_openai_agents
from pydantic import BaseModel, Field


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql:///enron_embeddings",
)
EMBEDDING_MODEL = os.environ.get("ENRON_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", "enron-email-agent")
MODAL_DATABASE_URL = "postgresql:///enron_embeddings?host=/var/run/postgresql"

logging.basicConfig(level=os.environ.get("ENRON_EVAL_LOG_LEVEL", "INFO"))
logger = logging.getLogger("enron_email_agent_eval")
_postgres_started = False
_postgres_start_lock = threading.Lock()
_eval_span: ContextVar[Any | None] = ContextVar("enron_eval_span", default=None)
setup_openai_agents(project_name=PROJECT_NAME)


class ModelParam(BaseModel):
    value: str = Field(default="gpt-5-mini", description="OpenAI model for the email-answering agent.")


def _parameter_value(parameters: dict[str, Any] | None, name: str, default: Any) -> Any:
    value = (parameters or {}).get(name, default)
    if isinstance(value, BaseModel):
        return getattr(value, "value", default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _start_postgres() -> None:
    global _postgres_started
    with _postgres_start_lock:
        if _postgres_started:
            return
        if shutil.which("start-enron-postgres"):
            logger.info("Starting bundled Postgres cluster with start-enron-postgres.")
            try:
                result = subprocess.run(
                    ["start-enron-postgres"],
                    capture_output=True,
                    text=True,
                    timeout=45,
                )
            except subprocess.TimeoutExpired as exc:
                diagnostics = _postgres_timeout_diagnostics(exc)
                logger.error("start-enron-postgres timed out. diagnostics=%s", diagnostics)
                _log_metadata("postgres_start_error", diagnostics)
                raise
            if result.returncode != 0:
                diagnostics = _postgres_diagnostics(result)
                logger.error("start-enron-postgres failed. diagnostics=%s", diagnostics)
                if _postgres_ready():
                    logger.warning("Postgres is reachable after start-enron-postgres failed; continuing.")
                else:
                    _log_metadata("postgres_start_error", diagnostics)
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        result.args,
                        output=result.stdout,
                        stderr=result.stderr,
                    )
        else:
            logger.info("No start-enron-postgres helper found; assuming external/local Postgres is already running.")
        if shutil.which("start-enron-postgres"):
            _ensure_runtime_database_user()
        _postgres_started = True


def _runtime_database_user() -> str:
    return os.environ.get("ENRON_DATABASE_USER") or getuser()


def _ensure_runtime_database_user() -> None:
    user = _runtime_database_user()
    if user == "postgres":
        return
    if not shutil.which("sudo") or not shutil.which("psql"):
        logger.warning("Cannot ensure Postgres role %r because sudo or psql is unavailable.", user)
        return

    sql = f"""
do $$
begin
  if not exists (select 1 from pg_roles where rolname = {_sql_literal(user)}) then
    execute 'create role ' || {_sql_literal(_sql_identifier(user))} || ' login';
  end if;
end
$$;
grant connect on database enron_embeddings to {_sql_identifier(user)};
grant usage on schema public to {_sql_identifier(user)};
grant select on all tables in schema public to {_sql_identifier(user)};
"""
    result = subprocess.run(
        [
            "sudo",
            "-u",
            "postgres",
            "psql",
            "--username",
            "postgres",
            "--dbname",
            "enron_embeddings",
            "--command",
            sql,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        diagnostics = {
            "runtime_user": user,
            "returncode": result.returncode,
            "stdout": _tail_text(result.stdout, 2000),
            "stderr": _tail_text(result.stderr, 2000),
        }
        logger.error("Failed to ensure runtime Postgres role. diagnostics=%s", diagnostics)
        _log_metadata("postgres_role_error", diagnostics)
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )


def _sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _postgres_ready() -> bool:
    if not shutil.which("pg_isready"):
        return False
    return (
        subprocess.run(
            ["pg_isready", "-q", "-h", "/var/run/postgresql", "-d", "enron_embeddings", "-U", "postgres"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).returncode
        == 0
    )


def _postgres_timeout_diagnostics(exc: subprocess.TimeoutExpired) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "command": list(exc.cmd) if isinstance(exc.cmd, list) else exc.cmd,
        "timeout": exc.timeout,
        "stdout": _tail_text(exc.stdout, 4000),
        "stderr": _tail_text(exc.stderr, 4000),
    }
    diagnostics.update(_postgres_probe_diagnostics())
    return diagnostics


def _postgres_diagnostics(start_result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "command": list(start_result.args),
        "returncode": start_result.returncode,
        "stdout": _tail_text(start_result.stdout, 4000),
        "stderr": _tail_text(start_result.stderr, 4000),
    }
    diagnostics.update(_postgres_probe_diagnostics())
    return diagnostics


def _postgres_probe_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for name, command in {
        "pg_isready": ["pg_isready", "-h", "/var/run/postgresql", "-d", "enron_embeddings", "-U", "postgres"],
        "pg_ctlcluster_status": ["pg_ctlcluster", "18", "main", "status"],
    }.items():
        if not shutil.which(command[0]):
            diagnostics[name] = {"error": f"{command[0]} not found"}
            continue
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=10)
            diagnostics[name] = {
                "returncode": probe.returncode,
                "stdout": _tail_text(probe.stdout, 2000),
                "stderr": _tail_text(probe.stderr, 2000),
            }
        except subprocess.TimeoutExpired as exc:
            diagnostics[name] = {
                "timeout": exc.timeout,
                "stdout": _tail_text(exc.stdout, 2000),
                "stderr": _tail_text(exc.stderr, 2000),
            }
    return diagnostics


def _tail_text(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value.strip()[-limit:]


def _database_url() -> str:
    if shutil.which("start-enron-postgres"):
        return f"{MODAL_DATABASE_URL}&user={quote(_runtime_database_user(), safe='')}"
    return DATABASE_URL


def _log_metadata(name: str, value: Any) -> None:
    try:
        span = _eval_span.get()
        if span is None:
            from braintrust import current_span

            span = current_span()
        span.log(metadata={name: value})
    except Exception:
        logger.debug("Unable to log Braintrust metadata %s.", name, exc_info=True)


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def _vector_literal(values: Any) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


@function_tool
def search_email_chunks(query: str, limit: int = 6) -> str:
    """Search Enron email chunks by semantic similarity and return source metadata plus excerpts."""
    db_url = _database_url()
    try:
        _start_postgres()
        safe_limit = max(1, min(int(limit), 20))
        logger.info("Searching Enron chunks. query=%r limit=%s database_url=%s", query, safe_limit, db_url)
        embedding = _embedding_model().encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        vector = _vector_literal(embedding)

        sql = """
            select
                e.source_file,
                e.sent_at,
                e.sender,
                e.recipients,
                e.subject,
                c.chunk_index,
                left(c.content, 1600) as excerpt,
                c.embedding <=> %s::vector as distance
            from enron_email_chunks c
            join enron_emails e on e.id = c.email_id
            order by c.embedding <=> %s::vector
            limit %s
        """
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vector, vector, safe_limit))
                rows = cur.fetchall()

        results = []
        for row in rows:
            source_file, sent_at, sender, recipients, subject, chunk_index, excerpt, distance = row
            results.append(
                {
                    "source_file": source_file,
                    "sent_at": sent_at.isoformat() if sent_at else None,
                    "sender": sender,
                    "recipients": list(recipients or [])[:12],
                    "subject": subject,
                    "chunk_index": chunk_index,
                    "excerpt": excerpt,
                    "distance": float(distance),
                }
            )
        _log_metadata(
            "search_email_chunks",
            {
                "query": query,
                "limit": safe_limit,
                "result_count": len(results),
                "database_url": db_url,
            },
        )
        logger.info("Enron search returned %s chunks.", len(results))
        return json.dumps(results, ensure_ascii=False)
    except Exception as exc:
        logger.exception("search_email_chunks failed.")
        _log_metadata(
            "search_email_chunks_error",
            {
                "query": query,
                "database_url": db_url,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


async def task(input: str, hooks) -> str:
    span_token = _eval_span.set(hooks.span)
    parameters = hooks.parameters or {}
    model = str(_parameter_value(parameters, "model", "gpt-5-mini"))
    query = str(input or "")
    try:
        if not query.strip():
            raise ValueError("Dataset input is required and should contain the question to ask about the emails.")
        hooks.metadata["model"] = model
        hooks.metadata["query"] = query
        hooks.metadata["database_url"] = _database_url()
        logger.info("Running Enron email agent. model=%s query=%r database_url=%s", model, query, _database_url())

        agent = Agent(
            name="Enron email analyst",
            model=model,
            instructions=(
                "Answer questions about the Enron email corpus. Use search_email_chunks before answering. "
                "Ground the answer in returned email excerpts and include source_file values for important claims. "
                "If the retrieved evidence is weak or absent, say that directly."
            ),
            tools=[search_email_chunks],
        )
        with hooks.span.start_span(name="openai_agents_runner"):
            result = await Runner.run(
                agent,
                input=f"Question: {query}",
                max_turns=6,
            )
        return str(result.final_output)
    finally:
        _eval_span.reset(span_token)


Eval(
    PROJECT_NAME,
    data=[
        {
            "input": "",
        }
    ],
    task=task,
    scores=[],
    parameters={
        "model": ModelParam,
    },
)
