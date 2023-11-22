import json


class ApiEndpoints:
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as f:
            data = json.load(f)
        self.SV_SENTIMENT = data.get("sv_sentiment")
        self.EN_SENTIMENT = data.get("en_sentiment")
        self.TOPIC = data.get("topic")
        self.TRANSLATION = data.get("translation")
        self.SUMMARIZE = data.get("summarize")
        self.DETECT_LANGUAGE = data.get("detect_language")
        self.MUL_SENTIMENT = data.get("mul_sentiment")
