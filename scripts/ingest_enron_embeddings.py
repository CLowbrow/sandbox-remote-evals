#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.parser import Parser
from email.utils import getaddresses, parsedate_to_datetime
from typing import Iterable

import psycopg
from psycopg.rows import tuple_row
from tqdm import tqdm


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class EmailRecord:
    source_file: str
    message_id: str | None
    date: datetime | None
    sender: str | None
    recipients: list[str]
    subject: str | None
    body: str
    raw_len: int
    body_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local embeddings for Enron emails and load them into pgvector."
    )
    parser.add_argument("--csv", default="data/emails.csv", help="Path to the Enron CSV.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", "postgresql:///enron_embeddings"),
        help="Postgres connection URL. Defaults to $DATABASE_URL or postgresql:///enron_embeddings.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name or local path.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Embedding device. Use cuda for the RTX 5080 path; auto uses CUDA when visible.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Emails to read before embedding chunks.")
    parser.add_argument("--embed-batch-size", type=int, default=256, help="Texts per model.encode call.")
    parser.add_argument("--chunk-chars", type=int, default=1600, help="Approximate characters per embedded chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Character overlap between chunks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional email limit for smoke tests.")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing enron tables before loading.")
    parser.add_argument(
        "--skip-vector-index",
        action="store_true",
        help="Skip creating the pgvector IVFFlat index after loading.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "Check WSL GPU access, NVIDIA drivers, and the installed PyTorch CUDA build."
        )
    return requested


def load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dimension = model.get_sentence_embedding_dimension()
    if not dimension:
        probe = model.encode(["dimension probe"], batch_size=1, convert_to_numpy=True)
        dimension = int(probe.shape[1])
    return model, int(dimension)


def create_schema(conn: psycopg.Connection, dimension: int, drop_existing: bool) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if drop_existing:
            cur.execute("DROP TABLE IF EXISTS enron_email_chunks")
            cur.execute("DROP TABLE IF EXISTS enron_emails")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS enron_emails (
                id bigserial PRIMARY KEY,
                source_file text NOT NULL UNIQUE,
                message_id text,
                sent_at timestamptz,
                sender text,
                recipients text[] NOT NULL DEFAULT '{}',
                subject text,
                body_sha256 text NOT NULL,
                raw_len integer NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS enron_email_chunks (
                id bigserial PRIMARY KEY,
                email_id bigint NOT NULL REFERENCES enron_emails(id) ON DELETE CASCADE,
                chunk_index integer NOT NULL,
                content text NOT NULL,
                embedding vector({dimension}) NOT NULL,
                UNIQUE (email_id, chunk_index)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS enron_emails_source_file_idx ON enron_emails (source_file)")
    conn.commit()


def create_vector_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS enron_email_chunks_embedding_ivfflat_idx
            ON enron_email_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )
        cur.execute("ANALYZE enron_emails")
        cur.execute("ANALYZE enron_email_chunks")
    conn.commit()


def parse_email(source_file: str, raw_message: str) -> EmailRecord:
    message = Parser(policy=policy.default).parsestr(raw_message)
    subject = clean_header(message.get("subject"))
    sender = clean_header(message.get("from"))
    recipients = extract_recipients(message)
    sent_at = parse_date(message.get("date"))
    body = extract_body(message)
    if not body.strip():
        body = raw_message
    return EmailRecord(
        source_file=source_file,
        message_id=clean_header(message.get("message-id")),
        date=sent_at,
        sender=sender,
        recipients=recipients,
        subject=subject,
        body=body,
        raw_len=len(raw_message),
        body_sha256=hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest(),
    )


def clean_header(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).split())
    return cleaned or None


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return parsed


def extract_recipients(message) -> list[str]:
    values: list[str] = []
    for header in ("to", "cc", "bcc"):
        values.extend(message.get_all(header, []))
    return [addr.lower() for _, addr in getaddresses(values) if addr]


def extract_body(message) -> str:
    if message.is_multipart():
        part = message.get_body(preferencelist=("plain", "html"))
        if part is not None:
            payload = part.get_content()
            return str(payload)
        return "\n".join(str(part.get_content()) for part in message.walk() if part.get_content_maintype() == "text")
    try:
        return str(message.get_content())
    except LookupError:
        payload = message.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(message.get_payload())


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start, end), normalized.rfind(" ", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
        if start < len(normalized) and normalized[start].isspace():
            start += 1
        else:
            start += step if start == end else 0
    return chunks


def embedding_text(record: EmailRecord, chunk: str) -> str:
    parts = [
        f"Subject: {record.subject or ''}",
        f"From: {record.sender or ''}",
        f"To: {', '.join(record.recipients[:20])}",
        "",
        chunk,
    ]
    return "\n".join(parts)


def vector_literal(values) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def csv_records(path: str, limit: int | None) -> Iterable[EmailRecord]:
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if limit is not None and index > limit:
                break
            yield parse_email(row["file"], row["message"])


def batched(iterable: Iterable[EmailRecord], size: int) -> Iterable[list[EmailRecord]]:
    batch: list[EmailRecord] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def insert_batch(
    conn: psycopg.Connection,
    model,
    records: list[EmailRecord],
    embed_batch_size: int,
    chunk_chars: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    email_rows = [
        (
            record.source_file,
            record.message_id,
            record.date,
            record.sender,
            record.recipients,
            record.subject,
            record.body_sha256,
            record.raw_len,
        )
        for record in records
    ]
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.executemany(
            """
            INSERT INTO enron_emails
                (source_file, message_id, sent_at, sender, recipients, subject, body_sha256, raw_len)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file) DO UPDATE SET
                message_id = EXCLUDED.message_id,
                sent_at = EXCLUDED.sent_at,
                sender = EXCLUDED.sender,
                recipients = EXCLUDED.recipients,
                subject = EXCLUDED.subject,
                body_sha256 = EXCLUDED.body_sha256,
                raw_len = EXCLUDED.raw_len
            """,
            email_rows,
        )
        cur.execute(
            "SELECT id, source_file FROM enron_emails WHERE source_file = ANY(%s)",
            ([record.source_file for record in records],),
        )
        ids_by_source = {source: email_id for email_id, source in cur.fetchall()}

    chunk_rows: list[tuple[int, int, str, str]] = []
    texts: list[str] = []
    metadata: list[tuple[int, int, str]] = []
    for record in records:
        email_id = ids_by_source[record.source_file]
        for chunk_index, chunk in enumerate(chunk_text(record.body, chunk_chars, chunk_overlap)):
            texts.append(embedding_text(record, chunk))
            metadata.append((email_id, chunk_index, chunk))

    if texts:
        embeddings = model.encode(
            texts,
            batch_size=embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        chunk_rows = [
            (email_id, chunk_index, chunk, vector_literal(embedding))
            for (email_id, chunk_index, chunk), embedding in zip(metadata, embeddings, strict=True)
        ]

    with conn.cursor() as cur:
        if ids_by_source:
            cur.execute("DELETE FROM enron_email_chunks WHERE email_id = ANY(%s)", (list(ids_by_source.values()),))
        if chunk_rows:
            with cur.copy(
                "COPY enron_email_chunks (email_id, chunk_index, content, embedding) FROM STDIN"
            ) as copy:
                for row in chunk_rows:
                    copy.write_row(row)
    conn.commit()
    return len(records), len(chunk_rows)


def main() -> None:
    args = parse_args()
    if args.chunk_chars <= 0:
        raise SystemExit("--chunk-chars must be greater than zero.")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
        raise SystemExit("--chunk-overlap must be non-negative and smaller than --chunk-chars.")
    if args.batch_size <= 0 or args.embed_batch_size <= 0:
        raise SystemExit("--batch-size and --embed-batch-size must be greater than zero.")

    device = resolve_device(args.device)
    print(f"Loading model {args.model!r} on {device}...", flush=True)
    model, dimension = load_model(args.model, device)
    print(f"Embedding dimension: {dimension}", flush=True)

    with psycopg.connect(args.database_url) as conn:
        create_schema(conn, dimension, args.drop_existing)
        total_emails = 0
        total_chunks = 0
        batches = batched(csv_records(args.csv, args.limit), args.batch_size)
        for records in tqdm(batches, desc="Embedding/loading batches", unit="batch"):
            email_count, chunk_count = insert_batch(
                conn,
                model,
                records,
                args.embed_batch_size,
                args.chunk_chars,
                args.chunk_overlap,
            )
            total_emails += email_count
            total_chunks += chunk_count
        if not args.skip_vector_index:
            print("Creating pgvector index and analyzing tables...", flush=True)
            create_vector_index(conn)
    print(f"Loaded {total_emails} emails and {total_chunks} chunks.", flush=True)


if __name__ == "__main__":
    main()
