#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
IMAGE="${VIDEO_EDITOR_IMAGE:-orin-video-editor:jp6}"
BASE_IMAGE="${VIDEO_EDITOR_BASE_IMAGE:-conf-summarizer:jetson-orin}"

exec docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --tag "${IMAGE}" \
  "${PROJECT_ROOT}"

