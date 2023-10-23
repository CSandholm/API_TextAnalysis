import json
from enum import Enum


class Endpoints(Enum):
    def __init__(self):
        with open("../configs/config_endpoints.json") as f:
            data = json.load(f)
        self.SV_SENTIMENT = data.get("sv_sentiment")
        self.EN_SENTIMENT = data.get("en_sentiment")
        self.TOPIC = data.get("topic")
        self.TRANSLATION = data.get("translation")
        self.SUMMARIZE = data.get("translation")
        self.DETECT_LANGUAGE = data.get("detect_language")
