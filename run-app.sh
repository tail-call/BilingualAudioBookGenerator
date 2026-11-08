#!/bin/bash
# This script is executed inside the Docker image.

pushd /root/llama.cpp
./run-qwen.sh &
LLM_PID=$!

pushd /root/audio.cpp
./run-supertonic.sh &
TTS_PID=$!

# Kill llama.cpp and audio.cpp on exit
trap "kill $BG_PID 2>/dev/null" EXIT INT TERM

# Run app
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8100
