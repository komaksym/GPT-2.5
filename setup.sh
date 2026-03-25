#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

git remote set-url origin git@github.com:komaksym/GPT-2.5.git
git config --global user.email "kovalmaksym2@gmail.com"
git config --global user.name "komaksym"

uv run --project training prek install \
  --hook-type pre-commit \
  --prepare-hooks
