import json
import torch

from app.api.ai.ABCProcedures import Procedure
from app.api.ai.ai_utils import languages
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM

with open("C:/Users/CharlesSandholm/PycharmProjects/AI_APIs/API_TextAnalysis/app/configs/config_ai_models.json") as f:
    config = json.load(f)

sv_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("sv_sentiment_model_path"))
sv_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_sentiment_model_path"))

en_sentiment_model = AutoModelForSequenceClassification.from_pretrained(config.get("en_sentiment_model_path"))
en_sentiment_tokenizer = AutoTokenizer.from_pretrained(config.get("en_sentiment_model_path"))

sv_summarize_text_model = AutoModelForSeq2SeqLM.from_pretrained(config.get("sv_summarize_text_path"))
sv_summarize_text_tokenizer = AutoTokenizer.from_pretrained(config.get("sv_summarize_text_path"))

detect_language_model = AutoModelForSequenceClassification.from_pretrained(config.get("detect_language_model_path"))
detect_language_tokenizer = AutoTokenizer.from_pretrained(config.get("detect_language_model_path"))


class DetectLanguageProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = detect_language_tokenizer.encode(_input, return_tensors='pt')
        result = detect_language_model(tokens)
        result_logits = result.logits
        result_index = int(torch.argmax(result_logits))
        result_list = languages.get_language_code(result_index)
        score = str(round(result_logits[0][result_index].item()))
        result_list.append(score)

        return result_list


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


class SvSummarizeTextProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = sv_summarize_text_tokenizer(_input, return_tensors="pt").input_ids
        outputs = sv_summarize_text_model.generate(input_ids=tokens, max_length=130, num_beams=5, num_return_sequences=1)
        result = sv_summarize_text_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result[0]
