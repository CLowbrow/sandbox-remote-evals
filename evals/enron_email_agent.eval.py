from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any

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
    if _postgres_started:
        return
    if shutil.which("start-enron-postgres"):
        logger.info("Starting bundled Postgres cluster with start-enron-postgres.")
        subprocess.run(["start-enron-postgres"], check=True)
    else:
        logger.info("No start-enron-postgres helper found; assuming external/local Postgres is already running.")
    _postgres_started = True


def _database_url() -> str:
    if shutil.which("start-enron-postgres"):
        return MODAL_DATABASE_URL
    return DATABASE_URL


def _log_metadata(name: str, value: Any) -> None:
    try:
        from braintrust import current_span

        current_span().log(metadata={name: value})
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
    parameters = hooks.parameters or {}
    model = str(_parameter_value(parameters, "model", "gpt-5-mini"))
    query = str(input or "")
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
    result = await Runner.run(
        agent,
        input=f"Question: {query}",
        max_turns=6,
    )
    return str(result.final_output)


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
