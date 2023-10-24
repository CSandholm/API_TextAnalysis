import json

from app.api.ai.ABCProcedures import Procedure
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

with open("C:/Users/CharlesSandholm/PycharmProjects/AI_APIs/API_TextAnalysis/app/configs/config_ai_models.json") as f:
    config = json.load(f)

sv_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("sv_sentiment_model_path"))
sv_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_sentiment_model_path"))

en_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("en_sentiment_model_path"))
en_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("en_sentiment_model_path"))


class SvSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = sv_sentiment_tokenizer.encode(_input, return_tensors="pt")
        result = sv_sentiment_model(tokens)
        output_np = result.logits[0].detach().numpy()
        output = softmax(output_np)
        return output


class EnSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = en_sentiment_tokenizer.encode(_input, return_tensors="pt")
        result = en_sentiment_model(tokens)
        output_np = result.logits[0].detach().numpy()
        output = softmax(output_np)
        return output
