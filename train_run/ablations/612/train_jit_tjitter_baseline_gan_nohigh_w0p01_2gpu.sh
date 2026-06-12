#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
TAG_SUFFIX="jit_tjitter_mu0p02_std0p02_gan_nohigh_w0p01"
JITTER_MEAN="0.02"
JITTER_STD="0.02"
GAN_NOHIGH_WEIGHT="0.01"
source "${DIR}/_common_jit_tjitter_2gpu.sh"
launch_jit_tjitter_2gpu "$@"
