import json

from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.api.ai.ABCProcedures import Procedure

with open("C:/Users/CharlesSandholm/PycharmProjects/AI_APIs/API_TextAnalysis/app/configs/config_ai_models.json") as f:
    config = json.load(f)

sv_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("sv_sentiment_model_path"))
sv_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_sentiment_model_path"))


class SvSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = sv_sentiment_model.encode(_input, return_tensors="pt")
        result = sv_sentiment_tokenizer(tokens)
        output_np = result.logits[0].detach().numpy()
        output = softmax(output_np)
        return output
