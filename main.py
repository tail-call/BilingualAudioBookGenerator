import json
from datetime import datetime
from functools import reduce
from glob import glob
from itertools import islice
from typing import Literal
from urllib.request import urlretrieve

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, ConfigDict, Field
from pydub import AudioSegment
from pypdf import PdfReader

from prompts import CAPTION_PROMPT, CLEANER_PROMPT
from qwen import generate_audio

reader = PdfReader("inkheart_RuLit_Me_789272.pdf")

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

def message_content_from(
    llm_response: ModelResponse | CustomStreamWrapper
) -> str | None:
    match llm_response:
        case ModelResponse(choices=choices):
            return choices[0].message.content
        case CustomStreamWrapper():
            return None

def process_book():
    with open("out.jsonl", "w") as out:
        page_number = 0
        batch_size = 5
        while True:
            time = datetime.now().strftime("%H:%M:%S")
            print(f"{time} Working with pages {page_number}..<{page_number + batch_size}")
            response = litellm.completion(
                model="openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
                api_base="http://localhost:8080/v1",
                api_key="not-needed",
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


def read_jsonl(filename):
    with open(filename) as jsonl:
        lines: list[str] = jsonl.readlines()
        objects = []
        for (i, line) in enumerate(lines):
            objects.append(json.loads(line))

        return objects

def generate_sentence():
    sentences = list(islice((x for x in read_jsonl("out.jsonl") if x["type"] == "sentence"), 5))

    # for (i, sentence) in enumerate(sentences):
    #     generate_speech(sentence["content"], f"sentence-{i}.wav")

    response = litellm.completion(
        model="openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M",
        api_base="http://localhost:8080/v1",
        api_key="not-needed",
        temperature=0.0,
        messages=[{
            "role": "system",
            "content": CAPTION_PROMPT.format(
                source_language="English",
                target_language="Russian",
            ),
        }, {
            "role": "user",
            "content": f"""
<context>
{" ".join([sentence["content"] for sentence in sentences])}
</context>

<target>
{sentences[3]["content"]}
</target>
            """
        }],
        response_format=TranslationOutput,
    )

    output = TranslationOutput.model_validate_json(
        message_content_from(llm_response=response) or ""
    )
    translation_pairs: list[str] = []
    for sentence in output.sentences:
        translation_pairs.append(f"{sentence.source} --- {sentence.target}")
        for word in sentence.uncommon_words or []:
            translation_pairs.append(f"{word.source} --- {word.target}")

    total = AudioSegment.empty()
    for i, pair in enumerate(translation_pairs):
        print(f"Generating speech for pair: {pair}")
        segment = generate_audio(pair, "Speak calmly with neutral emotion.")
        total += segment

    export_filename = "wowsuper.mp3"
    print(f"Writing to {export_filename}", end="...")
    total.export(export_filename, format="mp3")
    print("done.")

def ohman():
    generate_sentence()

    segments = [AudioSegment.from_wav(filename) for filename in glob("output/pair-*.wav")]
    total = AudioSegment.empty()
    for segment in segments:
        total += segment

    total.export("wow.mp3", format="mp3")

# if __name__ == "__main__":
#     generate_sentence()
