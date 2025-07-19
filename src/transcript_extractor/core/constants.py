"""Constants for transcript extraction models and configuration."""

# WhisperX supported models
WHISPER_MODELS = [
    "tiny",
    "base", 
    "small",
    "medium",
    "large-v2",
    "large-v3"
]

# Breeze ASR 25 model name
BREEZE_MODEL = "breeze-asr-25"

# All supported models
ALL_MODELS = WHISPER_MODELS + [BREEZE_MODEL]

# Model type detection
def is_breeze_model(model_name: str) -> bool:
    """Check if the given model name is a Breeze ASR model."""
    return model_name == BREEZE_MODEL

def is_whisper_model(model_name: str) -> bool:
    """Check if the given model name is a WhisperX model."""
    return model_name in WHISPER_MODELS