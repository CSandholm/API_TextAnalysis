import json

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM


class Models:
    def __init__(self):
        with open("app/configs/config_ai_models.json") as f:
            config = json.load(f)
        self.sv_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("sv_sentiment_model_path"))
        self.sv_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_sentiment_model_path"))
        self.sv_sentiment_max_position_embeddings = 512

        self.en_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("en_sentiment_model_path"))
        self.en_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("en_sentiment_model_path"))
        self.en_sentiment_max_position_embeddings = 514

        self.sv_summarize_text_model = AutoModelForSeq2SeqLM.from_pretrained(config.get("sv_summarize_text_path"))
        self.sv_summarize_text_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_summarize_text_path"))
        self.sv_summarize_text_self_max_position_embeddings = 1024

        self.detect_language_model = AutoModelForSequenceClassification.from_pretrained(config.get("detect_language_model_path"))
        self.detect_language_tokenizer = AutoTokenizer.from_pretrained(config.get("detect_language_model_path"))
        self.detect_language_max_position_embeddings = 512

        self.sv_en_language_model = AutoModelForSeq2SeqLM.from_pretrained(config.get("sv_en_translation_model_path"))
        self.sv_en_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_en_translation_model_path"))
        self.sv_en_max_position_embeddings = 512

        self.en_sv_language_model = AutoModelForSeq2SeqLM.from_pretrained(config.get("en_sv_translation_model_path"))
        self.en_sv_tokenizer = AutoTokenizer.from_pretrained(config.get("en_sv_translation_model_path"))
        self.en_sv_max_position_embeddings = 512
