#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VIDEO_EDITOR_IMAGE:-orin-video-editor:jp6}"
USER_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
HF_CACHE_DIR="${HF_CACHE_DIR:-${USER_HOME}/.cache/huggingface}"

usage() {
  cat <<'EOF'
Usage:
  ./run_docker.sh process VIDEO [--edit-only] [--config FILE] [--force]
  ./run_docker.sh plan VIDEO [--config FILE] [--force]
  ./run_docker.sh summarize OUTPUT_JOB_DIR [--model MODEL] [--config FILE]
  ./run_docker.sh transcript OUTPUT_JOB_DIR
  ./run_docker.sh test
EOF
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "${PROJECT_ROOT}/output" "${PROJECT_ROOT}/models/asr" "${HF_CACHE_DIR}"

common=(
  docker run --rm
  --runtime nvidia
  --network host
  --ipc host
  --user "$(id -u):$(id -g)"
  --volume "${PROJECT_ROOT}:/workspace"
  --volume "${HF_CACHE_DIR}:/data/models/huggingface"
  --env "PYTHONPATH=/workspace/src"
  --env "HOME=/tmp"
  --env "HF_HOME=/data/models/huggingface"
  --env "HUGGINGFACE_HUB_CACHE=/data/models/huggingface/hub"
  --env "FFMPEG_BIN=/usr/bin/ffmpeg"
  --env "FFPROBE_BIN=/usr/bin/ffprobe"
  --workdir /workspace
  --entrypoint python3
  "${IMAGE}"
)

command="$1"
shift
case "${command}" in
  plan|process)
    if [[ $# -eq 1 && "$1" == "--help" ]]; then
      exec "${common[@]}" -m local_video_editor "${command}" --help
    fi
    if [[ $# -lt 1 ]]; then usage; exit 2; fi
    source_path="$(realpath "$1")"
    shift
    source_dir="$(dirname "${source_path}")"
    source_name="$(basename "${source_path}")"
    extra=(--volume "${source_dir}:/input:ro")
    exec "${common[@]:0:${#common[@]}-1}" "${extra[@]}" "${IMAGE}" \
      -m local_video_editor "${command}" "/input/${source_name}" \
      --model-cache /data/models/huggingface/hub "$@"
    ;;
  summarize|transcript)
    if [[ $# -eq 1 && "$1" == "--help" ]]; then
      exec "${common[@]}" -m local_video_editor "${command}" --help
    fi
    if [[ $# -lt 1 ]]; then usage; exit 2; fi
    job_dir="$(realpath "$1")"
    shift
    case "${job_dir}" in
      "${PROJECT_ROOT}"/*) container_job="/workspace/${job_dir#"${PROJECT_ROOT}/"}" ;;
      *) echo "Job must be inside ${PROJECT_ROOT}" >&2; exit 2 ;;
    esac
    exec "${common[@]}" -m local_video_editor "${command}" "${container_job}" "$@"
    ;;
  test)
    exec "${common[@]}" -m unittest discover -s tests -v
    ;;
  *)
    usage
    exit 2
    ;;
esac
