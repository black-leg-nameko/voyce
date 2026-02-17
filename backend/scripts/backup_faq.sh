#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="${ROOT_DIR}/data/faq_ja.json"
BACKUP_DIR="${ROOT_DIR}/data/backups"
TS="$(date +%Y%m%d_%H%M%S)"
DST="${BACKUP_DIR}/faq_ja_${TS}.json"

mkdir -p "${BACKUP_DIR}"
cp "${SRC}" "${DST}"

echo "backup created: ${DST}"
