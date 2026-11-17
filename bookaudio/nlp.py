import json
from typing import Literal

import litellm

from pydantic import BaseModel, ConfigDict, Field
from bookaudio.constants import LLM_API_BASE, LLM_MODEL
from bookaudio.prompts import CAPTION_PROMPT, CLEANER_PROMPT
from bookaudio.util import load_jsonl_from_string


SentenceType = Literal[
    "title",
    "author",
    "chapterName",
    "sentence",
    "incompleteSentence",
]


class ParsedSentence(BaseModel):
    type: SentenceType
    content: str = Field(..., description="The text content of the sentence")


class TranslationOutput(BaseModel):
    sentences: list[SentencePair]


class UncommonWord(BaseModel):
    source: str
    target: str


class SentencePair(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str
    target: str
    uncommon_words: list[UncommonWord] | None = Field(
        default=None, alias="uncommonWords"
    )


class NLPService:
    async def translate_sentence(
        self,
        source_language: str,
        target_language: str,
        context_text: str,
        sentence_text: str,
    ) -> TranslationOutput:
        user_prompt = (
            f"<context>\n{context_text}\n</context>\n\n"
            f"<target>\n{sentence_text}\n</target>\n"
        )
        content = await self._ask_llm(
            CAPTION_PROMPT.format(
                source_language=source_language,
                target_language=target_language,
            ),
            user_prompt,
            response_format=TranslationOutput
        )
        content_json = json.loads(content)
        return TranslationOutput.model_validate(content_json)

    async def extract_sentences(self, text: str) -> list[ParsedSentence]:
        content = await self._ask_llm(CLEANER_PROMPT, text)
        return [
            ParsedSentence.model_validate(dictionary)
            for dictionary
            in load_jsonl_from_string(content)
        ]
    
    async def _ask_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        response = await litellm.acompletion(
            model=LLM_MODEL,
            api_base=LLM_API_BASE,
            api_key="not-needed",
            temperature=0.0,
            reasoning_effort="none",
            allowed_openai_params=["reasoning_effort"],
            response_format=response_format,
            messages=[{
                "role": "system", "content": system_prompt,
            }, {
                "role": "user", "content": user_prompt,
            }],
        )

        assert type(response) == litellm.ModelResponse, (
            "Unexpected response type from .acompletion call: "
            "did stream=True sneak in somewhere?"
        )

        return response.choices[0].message.content or ""