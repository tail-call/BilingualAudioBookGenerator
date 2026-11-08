#!/bin/bash
# This script must be copied to llama.cpp folder and ran from there
# Model source: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF

# echo commands
set -x

context_length=262144
model=unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
# model=unsloth/Qwen3.6-27B-MTP-GGUF

./build/bin/llama-server \
   --host 0.0.0.0 \
   -hf $model \
   -c $context_length \
   -fa on \
   -np 1 \
   --spec-type draft-mtp \
   --spec-draft-n-max 2
