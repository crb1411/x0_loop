#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TAG_SUFFIX="jit_tjitter_mu0p02_std0p025"
JITTER_MEAN="0.02"
JITTER_STD="0.025"
source "${DIR}/_common_jit_tjitter_2gpu.sh"
launch_jit_tjitter_2gpu "$@"
