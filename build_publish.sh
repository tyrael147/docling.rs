#!/bin/sh
# Build the CUDA 12.8 docling-serve image and tag it for the internal registry,
# in the background. Output goes to build_publish.log; the PID is printed.
#   ./build_publish.sh            # tag fix1
#   ./build_publish.sh fix2       # another tag
#   tail -f build_publish.log     # follow progress
set -eu

TAG="${1:-fix1}"
IMAGE="docling-rs-serve-cuda12:${TAG}"
REMOTE="artefacts.list.lu/docker-local/docling-rs-serve-cuda12:${TAG}"
LOG="build_publish.log"

# Run from the repo root (the Docker build context) wherever it is invoked from.
cd "$(dirname "$0")"

nohup sh -c "
  set -e
  echo \"[\$(date)] building ${IMAGE}\"
  docker build -f crates/docling-serve/Dockerfile --target serve-cuda12 -t '${IMAGE}' .
  docker image tag '${IMAGE}' '${REMOTE}'
  echo \"[\$(date)] done: ${REMOTE}\"
" > "$LOG" 2>&1 &

echo "build started in background (PID $!), logging to $(pwd)/${LOG}"
