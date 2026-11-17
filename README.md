# Bilingual Audio Book Generator

<!--For Zuri-->

A bilingual audio book generator for learning English. It uses LLM
and a TTS model to prepare a complete audio book. It has a web UI.

Stack: llama.cpp, audio.cpp, FastAPI, litellm, SQLAlchemy, PostgreSQL


## Run using Docker

```shell
# Run the app
docker compose up
# Run bash in the app's container
docker compose run --rm --entrypoint bash app
```

## Run locally

Run llama.cpp using the `run-qwen.sh` script, and audio.cpp using
the `server.json` config. You also need to have PostgreSQL running.
To initialize the database and create a `bookaudio_app` user,
use `db-init.sh` script.

To initialize $BOOKAUDIO_APP_PASSWORD I suggest doing something like this:

```shell
openssl rand -hex 32
```

Then you can run backend:

```shell
uvicorn bookaudio.app:app --host 0.0.0.0 --port 8100
```

If you ever need to change password for the `bookaudio_app` user in PostgreSQL:

```sql
ALTER USER bookaudio_app WITH PASSWORD 'new_strong_random_secret';
```

## AI Usage Disclaimer

The following large language models were used to assist with the code
generation:
- Qwen3.6-35B-A3B
- Gemini 3.8 Flash
- Gemini 4 Hydrogen

A human author reviewed, edited, and holds final responsibility for
the accuracy, originality, and integrity of the work.