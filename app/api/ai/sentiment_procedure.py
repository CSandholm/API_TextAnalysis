import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


class Sentiment:
    def __init__(self):
        with open("model_config.json", "r") as f:
            config = json.load(f)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.get("model_path"))
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path"))
    async def calc_sentiment(self, input):
        tokens = await self.tokenizer.encode(input, return_tensors="pt")
        result = await self.model(tokens)
        output_np = result.logits[0].detach().cpu().numpy()
        output = softmax(output_np)
        return output
