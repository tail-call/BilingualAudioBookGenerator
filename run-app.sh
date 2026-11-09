#!/bin/bash
# This script will run the app. It is meant to be executed inside
# the Docker container but you do you.

source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8100
