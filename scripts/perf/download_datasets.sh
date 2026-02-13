#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="assets/bench"
RAW_DIR="$DATA_DIR/raw"
MANIFEST="$DATA_DIR/MANIFEST.csv"
SOURCE_LIST="${1:-}"

mkdir -p "$RAW_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  cat > "$MANIFEST" <<'CSV'
dataset_id,source_url,license,checksum_sha256,vertex_count,face_count,notes
CSV
fi

# Keep local bunny as a baseline fixture.
cp -f assets/bunny.obj "$RAW_DIR/bunny.obj"
BUNNY_SHA="$(shasum -a 256 "$RAW_DIR/bunny.obj" | awk '{print $1}')"
if ! rg -q '^bunny_local,' "$MANIFEST"; then
  echo "bunny_local,file://assets/bunny.obj,LOCAL_ASSET,$BUNNY_SHA,2503,4968,Repository baseline fixture" >> "$MANIFEST"
fi

if [[ -z "$SOURCE_LIST" ]]; then
  echo "No source list provided."
  echo "Provide a CSV file with lines: dataset_id,url,license"
  echo "Example: ./scripts/perf/download_datasets.sh datasets.csv"
  exit 0
fi

while IFS=, read -r dataset_id url license; do
  [[ -z "$dataset_id" || -z "$url" || -z "$license" ]] && continue

  out_path="$RAW_DIR/${dataset_id}.obj"
  curl -L "$url" -o "$out_path"
  sha="$(shasum -a 256 "$out_path" | awk '{print $1}')"

  if ! rg -q "^${dataset_id}," "$MANIFEST"; then
    echo "${dataset_id},${url},${license},${sha},,,Downloaded via script" >> "$MANIFEST"
  fi

done < "$SOURCE_LIST"

echo "Updated manifest: $MANIFEST"
