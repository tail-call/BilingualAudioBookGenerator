import io
from functools import cache

import soundfile as sf
import torch
from pydub import AudioSegment
from qwen_tts import Qwen3TTSModel

# model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model_id = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"

# We cache this so model doesn't load every time module is imported
@cache
def get_model() -> Qwen3TTSModel:
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        attn_implementation="eager",
        # The only possible arguments are `attn_implementation="eager"`,
        #  `"attn_implementation=flash_attention_3"`,
        #  `"attn_implementation=flash_attention_2"`, `"attn_implementation=sdpa".
    )
    return model

# batch inference (LOOKS USEFUL)
# wavs, sr = model.generate_custom_voice(
#     text=[
#         "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
#         "She said she would be here by noon."
#     ],
#     language=["Chinese", "English"],
#     speaker=["Vivian", "Ryan"],
#     instruct=["", "Very happy."]
# )
# sf.write("output_custom_voice_1.wav", wavs[0], sr)
# sf.write("output_custom_voice_2.wav", wavs[1], sr)

test_text = "Many years later, Meggie had only to close her eyes and she could still hear it, like tiny fingers tapping on the windowpane.",

def generate_audio(
    text: str,
    instruct: str,
    language: str,
    ) -> AudioSegment:
    # audio, samplerate = model.generate_custom_voice(
    #     text=text,
    #     language=language,
    #     speaker="Aiden",
    #     instruct=instruct,
    # )
    model = get_model()
    audio, samplerate = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio="ref_audio.mp3",
        ref_text="Many years later, Meggie had only to close her eyes (много лет спустя, Мегги стоило только закрыть глаза), and she could still hear it (и она всё ещё могла слышать это), like tiny fingers tapping on the windowpane (словно крошечные пальцы постукивали по оконному стеклу / windowpane - оконное стекло)",
    )
    buffer = io.BytesIO()

    with sf.SoundFile(
        buffer, mode='w', samplerate=samplerate, channels=1, format='WAV'
    ) as f:
        f.title = text
        f.artist = model_id
        f.comment = instruct

        # Write audio data into the buffer
        f.write(audio[0])

    buffer.seek(0)

    return AudioSegment.from_wav(buffer)
