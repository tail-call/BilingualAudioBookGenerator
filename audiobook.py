import json
import math
from datetime import datetime
from functools import reduce
from typing import Literal

import httpx
import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict, Field
from pydub import AudioSegment
from pypdf import PdfReader

from prompts import CAPTION_PROMPT, CLEANER_PROMPT
from util import read_jsonl


class UncommonWord(BaseModel):
    source: str
    target: str

class SentencePair(BaseModel):
    # Allows validation using both the Python field name and the JSON alias
    model_config = ConfigDict(populate_by_name=True)

    source: str
    target: str
    uncommon_words: list[UncommonWord] | None = Field(
        default=None,
        alias="uncommonWords"
    )

class TranslationOutput(BaseModel):
    sentences: list[SentencePair]


def batch_indices(items_length: int, i: int, batch_size: int) -> range:
    assert batch_size <= items_length
    assert i >= 0
    assert i < items_length
    if i < batch_size / 2:
        return range(batch_size)
    elif i >= items_length - batch_size / 2:
        return range(items_length - batch_size, items_length)
    else:
        start = math.ceil(i - batch_size / 2)
        return range(start, start + batch_size)


def message_content_from(
    llm_response: ModelResponse | CustomStreamWrapper
) -> str | None:
    match llm_response:
        case ModelResponse(choices=choices):
            return choices[0].message.content
        case CustomStreamWrapper():
            return None

def time_now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def extract_sentences(
    in_filename: str,
    out_filename: str,
    batch_size: int = 5,
):
    reader = PdfReader(in_filename)

    with open(out_filename, "w") as out:
        page_number = 0
        while True:
            print(f"{time_now()} Working with pages {page_number}..<{page_number + batch_size}")
            response = litellm.completion(
                model="openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
                api_base="http://localhost:8080/v1",
                api_key="not-needed",
                temperature=0.0,
                reasoning_effort="none",
                allowed_openai_params=['reasoning_effort'],
                messages=[{
                    "role": "system",
                    "content": CLEANER_PROMPT,
                }, {
                    "role": "user",
                    "content": reduce(
                        lambda a, b: a + b,
                        (page.extract_text() for page in reader.pages[
                            page_number:page_number+batch_size
                        ])
                    )
                }]
            )
            content = message_content_from(llm_response=response)
            if content:
                out.write(content)
            else:
                print(
                    "WARNING: empty content when processing pages "
                    f"{page_number}:{page_number+batch_size}"
                )

            page_number += batch_size
    print("Done writing.")


def generate_annotations(
    in_filename: str,
    out_filename: str,
    source_language = "English",
    target_language = "Russian",
):

    # XXX: This function doesn't produce a valid JSONL as the JSONs it generates
    # span multiple lines! What a pity
    sentences = [x for x in read_jsonl(in_filename) if x["type"] == "sentence"]
    with open(out_filename, "w") as file:
        for (i, sentence) in enumerate(sentences):
            print(f"{time_now()} Working with sentence {i+1}/{len(sentences)}")
            context = [sentences[j] for j in batch_indices(len(sentences), i, 5)]
            response = litellm.completion(
                model="openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
                api_base="http://localhost:8080/v1",
                api_key="not-needed",
                temperature=0.0,
                reasoning_effort="none",
                allowed_openai_params=['reasoning_effort'],
                messages=[{
                    "role": "system",
                    "content": CAPTION_PROMPT.format(
                        source_language=source_language,
                        target_language=target_language,
                    ),
                }, {
                    "role": "user",
                    "content": f"""
            <context>
            {" ".join([sentence["content"] for sentence in context])}
            </context>

            <target>
            {sentence["content"]}
            </target>
                    """
                }],
                response_format=TranslationOutput,
            )
            content = message_content_from(response) or "// No content //"
            file.write(content + "\n")


def generate_audio(text, lang: Literal["ru", "en"]):
    print(f"Generating audio ({lang}): {text}")
    try:
        BASE_URL = "http://localhost:8919"

        response = httpx.post(
            BASE_URL + "/v1/audio/speech",
            content=json.dumps({
                'model': 'supertonic-3',
                'input': text,
                'language': lang
            })
        )

        return AudioSegment(response.content)
    except Exception as e:
        print("!!! EXCEPTION", e)
        return AudioSegment.from_file("error.flac")


def generate_tts(in_filename: str):
    with open(in_filename) as f:
        data = json.load(f)

    annotated_sentences = [
        TranslationOutput.model_validate(datum) for datum in data
    ]

    source_lang = "en"
    target_lang = "ru"

    seq = annotated_sentences
    for i, annotated_sentence in enumerate(seq):
        export_filename = f"output/gputonic-{i:05d}.mp3"
        print(f"# {time_now()} Sentence {i+1}/{len(seq)}")

        total = AudioSegment.empty()
        for sentence in annotated_sentence.sentences:
            total += generate_audio(sentence.source, source_lang)
            total += AudioSegment.silent(duration=200)
            print(f"Generating audio: {(sentence.target, target_lang)}")
            total += generate_audio(sentence.target, target_lang)
            for word in sentence.uncommon_words or []:
                total += AudioSegment.silent(duration=200)
                print(f"Generating audio: {(word.source, source_lang)}")
                total += generate_audio(word.source, source_lang)
                print(f"Generating audio: {(word.target, target_lang)}")
                total += generate_audio(word.target, target_lang)
            total += AudioSegment.silent(duration=400)

        total.export(export_filename, format="mp3")
        print(f"Written to {export_filename}")


if __name__ == "__main__":
    extract_sentences("pdf/deepwork.pdf", "scratch/deepwork.jsonl")
    generate_annotations("scratch/deepwork.jsonl", "scratch/deepwork-annotated.json")
    generate_tts("scratch/deepwork-annotated.json")
