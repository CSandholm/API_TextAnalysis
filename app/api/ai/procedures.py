import torch

from app.api.ai.ABCProcedures import Procedure
from app.api.ai.ai_utils import languages
from app.api.api_utils.models_handler import Models
from scipy.special import softmax

models = Models()


class TranslationProcedure:
    async def run_procedure(self, _input, src_lang):
        if src_lang != "sv":
            input_ids = models.en_sv_tokenizer(_input, return_tensors="pt").input_ids
            outputs = models.en_sv_language_model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
            result = models.en_sv_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translation = result[0]
            return translation
        else:
            input_ids = models.sv_en_tokenizer(_input, return_tensors="pt").input_ids
            outputs = models.sv_en_language_model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
            result = models.sv_en_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translation = result[0]
            return translation


class DetectLanguageProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = models.detect_language_tokenizer.encode(_input, return_tensors='pt')
        result = models.detect_language_model(tokens)
        result_logits = result.logits
        result_index = int(torch.argmax(result_logits))
        result_list = languages.get_language_code(result_index)
        score = str(round(result_logits[0][result_index].item()))
        result_list.append(score)

        return result_list


class SvSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = models.sv_sentiment_tokenizer.encode(_input, return_tensors="pt")
        result = models.sv_sentiment_model(tokens)
        output_np = result.logits[0].detach().numpy()
        output = softmax(output_np)
        return output


class EnSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = models.en_sentiment_tokenizer.encode(_input, return_tensors="pt")
        result = models.en_sentiment_model(tokens)
        output_np = result.logits[0].detach().numpy()
        output = softmax(output_np)
        return output


class SvSummarizeTextProcedure(Procedure):
    async def run_procedure(self, _input):
        tokens = models.sv_summarize_text_tokenizer(_input, return_tensors="pt").input_ids
        outputs = models.sv_summarize_text_model.generate(input_ids=tokens, max_length=130, num_beams=5, num_return_sequences=1)
        result = models.sv_summarize_text_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result[0]
