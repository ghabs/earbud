import enum

class WhisperModelSize(enum.Enum):
    TINY = 'tiny'
    BASE = 'base'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'

class WhisperModelType(enum.Enum):
    WHISPER = 'Whisper'
    WHISPER_CPP = 'Whisper.cpp'
    HUGGING_FACE = 'Hugging Face'

class EmbeddingsModelType(enum.Enum):
    OPEN_AI = 'OpenAI'
    HUGGING_FACE = 'Hugging Face'