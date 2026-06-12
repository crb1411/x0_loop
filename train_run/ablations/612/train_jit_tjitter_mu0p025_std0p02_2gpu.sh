#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TAG_SUFFIX="jit_tjitter_mu0p025_std0p02"
JITTER_MEAN="0.025"
JITTER_STD="0.02"
source "${DIR}/_common_jit_tjitter_2gpu.sh"
launch_jit_tjitter_2gpu "$@"
