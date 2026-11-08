# BookAudio

Stack: llama.cpp, audio.cpp, FastAPI, litellm, SQLAlchemy, PostgreSQL

For Zuri

Run backend:

```shell
uvicorn app:app --host 0.0.0.0 --port 8100
```

Docker stuff:

```
docker build -t bookaudio . # Build
docker run --gpus all -p 8100:8100 bookaudio # Run
docker run --gpus all -it bookaudio bash # Run bash
```

## Adding a user and a database to Postgres

```sql
CREATE USER bookaudio_app WITH ENCRYPTED PASSWORD 'strong_random_secret';
CREATE DATABASE bookaudio;
GRANT CONNECT ON DATABASE bookaudio TO bookaudio_app;

GRANT ALL PRIVILEGES ON DATABASE bookaudio TO bookaudio_app;
\c bookaudio
GRANT ALL ON SCHEMA public TO bookaudio_app;
```

To obtain a `'strong_random_secret'`, try using this command:

```shell
openssl rand -hex 32
```

If you ever need to change password:

```sql
ALTER USER bookaudio_app WITH PASSWORD 'new_strong_random_secret';
```

## BUGS

- [ ] Sentencizer producer sometimes glues two JSON objects on a single line
- [ ] Had a guy like this:

```
# 23:56:24 Sentence 1632/2791
Written to output/gputonic-1631.mp3
```
It was empty. Produced by this object:

```
  {
    "sentences": []
  },
```

Which was generated from this line:

```
{ "type": "sentence", "content": "[emphasis mine]" }
```

Gotta gotta gotta do something I guess

## TASKS

- [ ] Detect that llama.cpp and audio.cpp are running.
- [ ] Langfuse thing
- [ ] Try prefixing small sentences with dot
- [ ] Chapter names are missing in the annotated text
- [ ] Rewrite numbers as words
- [ ] Replace signs (+, -, =) with words
- [ ] Deduplicate keywords
- [ ] Process sentences where ['source'] == ['target']
- [ ] Extract sentences using an NLP model?
- [ ] Stress marks
- [ ] vLLM? SGLang?
- [x] Backend

## IDEAS

- [ ] Use the words frequency dictionary to estimate uncommon words instead of trying to predict them with an LLM.
- [ ] Use a simpler NLP model for sentence extraction.
- [ ] Propose exercises after each chapter



- https://github.com/Den4ikAI/ruaccent
- https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf

## Concat files in ffmpeg

```
ffmpeg -f concat -safe 0 -i filelist.txt -c copy deepwork.mp3
```


## AI Usage Disclaimer

The following large language models were used to assist with code generation:
- Qwen3.6-35B-A3B
- Gemini 3.8 Flash
