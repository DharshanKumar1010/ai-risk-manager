#!/usr/bin/env bash
# Fetches trained model artifacts (models/registry.json + models/artifacts/*) from the
# riskiq-models-v1 GitHub Release before uvicorn starts. Needed because trained weights are
# gitignored build outputs (see .gitignore's "Weights are build outputs" note) -- Render builds
# from the git tree alone, unlike docker-compose's local ./models:/models:ro bind mount, so
# there is nothing under models/ for app.core.serving.ModelBundle.load to read unless this
# script puts it there first.
#
# Idempotent by design: re-running with the artifacts already present is a no-op, so this is
# safe to prepend to the start command on every boot, not just the first one.
set -euo pipefail

RELEASE_URL="https://github.com/DharshanKumar1010/ai-risk-manager/releases/download/v1.0-models/riskiq-models-v1.zip"

# Resolved from this script's own location, not the caller's working directory -- Render's
# "Root Directory" setting decides what the start command's cwd is, and this must work
# regardless of what that is set to.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Settings.models_dir defaults to repo-root/models (app/config.py's DEFAULT_MODELS_DIR) but is
# env-overridable -- docker-compose sets it to /models for the container layout. Honouring the
# same env var here keeps this script correct under either layout instead of hardcoding one.
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
REGISTRY_PATH="$MODELS_DIR/registry.json"

if [[ -f "$REGISTRY_PATH" ]]; then
    echo "download_models.sh: $REGISTRY_PATH already exists, skipping download"
    exit 0
fi

echo "download_models.sh: fetching model artifacts from $RELEASE_URL"

TMP_ZIP="$(mktemp -t riskiq-models-XXXXXX.zip)"
cleanup() {
    rm -f "$TMP_ZIP"
}
trap cleanup EXIT

if ! curl -fsSL --retry 3 --retry-connrefused -o "$TMP_ZIP" "$RELEASE_URL"; then
    echo "download_models.sh: ERROR failed to download $RELEASE_URL" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"

if ! unzip -oq "$TMP_ZIP" -d "$MODELS_DIR"; then
    echo "download_models.sh: ERROR failed to extract $TMP_ZIP into $MODELS_DIR" >&2
    exit 1
fi

if [[ ! -f "$REGISTRY_PATH" ]]; then
    echo "download_models.sh: ERROR extraction completed but $REGISTRY_PATH is missing" >&2
    exit 1
fi

echo "download_models.sh: model artifacts ready at $MODELS_DIR"
