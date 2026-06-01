#!/usr/bin/env bash
# 00_install_deps.sh — system + Python deps for FabricCrypt.
# Tested on Ubuntu 24.04. Read before running.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[install] system deps (sudo) ..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential gcc python3 python3-venv python3-pip \
    linux-tools-common linux-tools-generic

echo "[install] python venv ..."
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "[install] compiling C helpers ..."
( cd src/signature && \
  gcc -O2 -march=native -pthread tsc_inter_core.c -o tsc_inter_core && \
  gcc -O2 -march=native -pthread cacheline_pingpong.c -o cacheline_pingpong )

echo "[install] DONE. Activate with:  source venv/bin/activate"
