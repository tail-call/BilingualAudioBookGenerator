#!/bin/bash
# This script will be copied to a Docker container to be run from there.
# Model source: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF

# echo commands
set -x

context_length=262144
model=/root/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
mmproj=/root/models/mmproj-BF16.gguf

./build/bin/llama-server \
   --host 0.0.0.0 \
   -m $model \
   -mm $mmproj \
   -c $context_length \
   -fa on \
   -np 1 \
   --spec-type draft-mtp \
   --spec-draft-n-max 2
