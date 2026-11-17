import asyncio
import io
from pathlib import Path
from typing import Literal

import httpx
from pydub import AudioSegment

from bookaudio.constants import TTS_BASE_URL
from bookaudio.nlp import TranslationOutput


async def generate_audio_clip(
    client: httpx.AsyncClient, text: str, lang: Literal["ru", "en"]
) -> AudioSegment:
    """Async call to the TTS engine."""
    resp = await client.post(
        f"{TTS_BASE_URL}/v1/audio/speech",
        json={"model": "supertonic-3", "input": text, "language": lang},
        timeout=120.0,
    )
    resp.raise_for_status()
    # Decode audio in worker thread to prevent event-loop blocking
    return await asyncio.to_thread(AudioSegment.from_file, io.BytesIO(resp.content))


async def render_sentence_audio(
    translation_dict: dict, export_path: Path
) -> None:
    """Generate and concatenate speech for a single annotated sentence."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    translation = TranslationOutput.model_validate(translation_dict)

    async with httpx.AsyncClient() as client:
        total = AudioSegment.empty()

        for sentence in translation.sentences:
            src_audio = await generate_audio_clip(client, sentence.source, "en")
            total += src_audio
            total += AudioSegment.silent(duration=200)

            tgt_audio = await generate_audio_clip(client, sentence.target, "ru")
            total += tgt_audio

            for word in sentence.uncommon_words or []:
                total += AudioSegment.silent(duration=200)
                total += await generate_audio_clip(client, word.source, "en")
                total += await generate_audio_clip(client, word.target, "ru")

            total += AudioSegment.silent(duration=400)

        # Export audio segment to disk asynchronously
        await asyncio.to_thread(total.export, str(export_path), format="mp3")


async def concat_files_with_ffmpeg(
    final_destination: Path, part_files: list[Path]
) -> None:
    """Glues part files using ffmpeg concat demuxer."""
    final_destination.parent.mkdir(parents=True, exist_ok=True)

    manifest_file = final_destination.with_suffix(".txt")
    with open(manifest_file, "w") as f:
        for p in part_files:
            f.write(f"file '{p.resolve()}'\n")

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_file),
            "-c",
            "copy",
            str(final_destination),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with code {proc.returncode}: {stderr.decode()}")
    finally:
        manifest_file.unlink(missing_ok=True)
