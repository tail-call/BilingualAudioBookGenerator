import os


LLM_API_BASE = os.getenv("LLM_API_BASE", "http://localhost:8080/v1")
LLM_MODEL = os.getenv(
    "LLM_MODEL", "openai/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M"
)
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8919")