#!/usr/bin/env bash
set -euo pipefail

DATABASE_URL="${DATABASE_URL:-postgresql:///enron_embeddings}"
OUTPUT="${1:-dumps/enron_embeddings.dump}"

mkdir -p "$(dirname "$OUTPUT")"
pg_dump \
  --format=custom \
  --compress=9 \
  --no-owner \
  --no-privileges \
  --dbname="$DATABASE_URL" \
  --file="$OUTPUT"

echo "Wrote $OUTPUT"

