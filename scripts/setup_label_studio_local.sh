#!/usr/bin/env bash
# Local Label Studio labeling with author-aligned DD-VQA frames (from tarkin).
#
# Usage:
#   bash scripts/setup_label_studio_local.sh prepare
#   bash scripts/setup_label_studio_local.sh start
#   bash scripts/setup_label_studio_local.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/label_studio/data"
FRAMES="$DATA/frames"
LS_ENV="${HOME}/.local/share/label-studio/.env"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

cmd="${1:-prepare}"

write_ls_env() {
  mkdir -p "$(dirname "$LS_ENV")"
  # Ensure flags persist across restarts (Label Studio also reads this .env).
  touch "$LS_ENV"
  # Remove previous keys then append.
  grep -vE '^(LABEL_STUDIO_)?LOCAL_FILES_(SERVING_ENABLED|DOCUMENT_ROOT)=' "$LS_ENV" > "${LS_ENV}.tmp" || true
  mv "${LS_ENV}.tmp" "$LS_ENV"
  {
    echo "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
    echo "LOCAL_FILES_SERVING_ENABLED=true"
    echo "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$DATA"
    echo "LOCAL_FILES_DOCUMENT_ROOT=$DATA"
  } >> "$LS_ENV"
  echo "==> Wrote local-files settings to $LS_ENV"
  echo "    DOCUMENT_ROOT=$DATA"
}

prepare_tasks() {
  if [[ ! -d "$FRAMES" ]]; then
    echo "ERROR: missing author frames at $FRAMES"
    echo "Copy from tarkin into label_studio/data/ first."
    exit 1
  fi
  for f in train.jsonl val.jsonl test.jsonl landmarks.jsonl; do
    if [[ ! -f "$DATA/$f" && ! -f "$DATA/${f/.jsonl/_loc.jsonl}" ]]; then
      echo "WARNING: missing $DATA/$f"
    fi
  done

  echo "==> Building tasks (global question only: 1 task per image)"
  "$PYTHON" "$ROOT/scripts/prepare_label_studio_ddvqa.py" \
    --ddvqa-dir "$DATA" \
    --landmarks-jsonl "$DATA/landmarks.jsonl" \
    --image-root "$FRAMES" \
    --document-root "$DATA" \
    --prefer-loc \
    --only-global-question \
    --output "$ROOT/label_studio/tasks_ddvqa_global.json"

  write_ls_env

  cat <<EOF

==> Next steps in the Label Studio UI (required for images to load):

1. Open the project
2. Settings → Cloud Storage → Add Source Storage
3. Storage Type: Local files
4. Absolute local path:
   $FRAMES
5. Leave "Treat every bucket object as a source file" ON (or Import method = Files)
6. Click Check Connection → Add Storage
   (Do NOT Sync if you already imported tasks — sync would create
    duplicate image-only tasks. The storage is only needed so
    /data/local-files/ is allowed to serve frames.)

Import file (global real/fake only, ~2968 unique images):
   $ROOT/label_studio/tasks_ddvqa_global.json
   Pre-boxes come as predictions (tasks show 0 annotations until you Submit).

After import, enable in the project:
   Settings → Machine Learning →
     "Show predictions to annotators in the Label Stream and Quick View"
   Then open Label All Tasks: boxes are copied into an editable annotation.
   Submit = human-reviewed (progress advances). Skip unfinished = stay at 0.

Do NOT use Actions → Create Annotations From Predictions on all tasks
(that would mark everything done again).

Region-specific questions are NOT imported; after labeling, boxes will
be propagated to eyes/nose/mouth/... Qs by lexical matching.

Labeling config:
   $ROOT/label_studio/ddvqa_loc_config.xml

Attention overlays (optional helper for labeling):
   python scripts/generate_ls_attention_overlays.py   # GPU recommended
   python scripts/patch_ls_attention_field.py \\
       --sqlite ~/.local/share/label-studio/label_studio.sqlite3 \\
       --project-title DD-VQA-Loc
   # Then paste updated ddvqa_loc_config.xml in Settings → Labeling Interface.
   # Local Storage / DOCUMENT_ROOT must be label_studio/data (not only frames/)
   # so /data/local-files/?d=attention/... resolves.
EOF
}

start_ls() {
  write_ls_env
  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
  export LOCAL_FILES_SERVING_ENABLED=true
  export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$DATA"
  export LOCAL_FILES_DOCUMENT_ROOT="$DATA"

  echo "==> DOCUMENT_ROOT=$DATA"
  echo "==> After start: add Local Storage pointing to:"
  echo "    $FRAMES"
  if [[ -x "$ROOT/venv/bin/label-studio" ]]; then
    exec "$ROOT/venv/bin/label-studio" start
  elif command -v label-studio >/dev/null 2>&1; then
    exec label-studio start
  else
    echo "label-studio not found"
    exit 1
  fi
}

case "$cmd" in
  prepare) prepare_tasks ;;
  start) start_ls ;;
  env) write_ls_env ;;
  all)
    prepare_tasks
    start_ls
    ;;
  *)
    echo "Usage: $0 {prepare|start|env|all}"
    exit 1
    ;;
esac
