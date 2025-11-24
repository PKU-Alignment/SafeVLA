#!/usr/bin/env bash
set -euo pipefail
REQ=./requirements.txt

while IFS= read -r pkg; do
    [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
    echo ">>> $pkg"
    if ! pip install --no-deps --no-build-isolation "$pkg"; then
        echo "[warn] $pkg install failed, continue" >&2
    fi
done < "$REQ"