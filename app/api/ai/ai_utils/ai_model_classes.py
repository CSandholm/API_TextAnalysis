from abc import ABC, abstractmethod
import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM


class AiModel(ABC):
    @abstractmethod
    def __init__(self):
        pass


class SvSentimentModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.get("sv_sentiment_model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("sv_sentiment_model_path"))
        self.max_position_embeddings = 512


class EnSentimentModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.get("en_sentiment_model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("en_sentiment_model_path"))
        self.max_position_embeddings = 514


class SvSummarizeTextModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.get("sv_summarize_text_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("sv_summarize_text_path"))
        self.max_position_embeddings = 1024


class DetectLanguageModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.get("detect_language_model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("detect_language_model_path"))
        self.max_position_embeddings = 512


class SvEnTranslationModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.get("sv_en_translation_model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("sv_en_translation_model_path"))
        self.max_position_embeddings = 512


class EnSvTranslationModel(AiModel):
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.get("en_sv_translation_model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("en_sv_translation_model_path"))
        self.max_position_embeddings = 512
