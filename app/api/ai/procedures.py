import json
import logging
import re

import torch

from app.api.ai.ABCProcedures import Procedure
from app.api.ai.ai_utils import languages
from app.api.ai.ai_utils.models_handler import Models
from scipy.special import softmax

logger = logging.getLogger(__name__)
models = Models()


class TopicProcedure(Procedure):
    async def run_procedure(self, _input):

        tokens = re.findall(r'\b\w+\b', _input.lower())
        mapped_tokens = []

        vocabulary = self.get_vocabulary()

        for token in tokens:
            for common_word, synonyms in vocabulary.items():
                if common_word not in mapped_tokens and token in synonyms:
                    mapped_tokens.append(common_word)
                    logger.info(f"Added: {common_word}")
                    break

        return mapped_tokens

    def get_vocabulary(self):
        with open("app/api/ai/ai_utils/topic_vocabulary.json", 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        return vocabulary


class TranslationProcedure:
    async def run_procedure(self, _input, src_lang):
        logger.info("Translation Procedure")
        try:
            if src_lang != "sv":
                logger.info("Translating English to Swedish")
                input_ids = models.en_sv_tokenizer(_input, return_tensors="pt").input_ids
                outputs = models.en_sv_language_model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
                result = models.en_sv_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translation = result[0]
                logger.info("Return Translation Procedure Result")
                return translation

            else:
                logger.info("Translating Swedish to English")
                input_ids = models.sv_en_tokenizer(_input, return_tensors="pt").input_ids
                outputs = models.sv_en_language_model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1)
                result = models.sv_en_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translation = result[0]
                logger.info("Return Translation Procedure Result")
                return translation

        except Exception as e:
            logger.warning(f"Translation procedure failed: {e}")
            return None


class DetectLanguageProcedure(Procedure):
    async def run_procedure(self, _input):
        try:
            logger.info("Detect Language Procedure")
            tokens = models.detect_language_tokenizer.encode(_input, return_tensors='pt')
            result = models.detect_language_model(tokens)
            result_logits = result.logits
            result_index = int(torch.argmax(result_logits))
            result_list = languages.get_language_code(result_index)
            score = str(round(result_logits[0][result_index].item()))
            result_list.append(score)
            logger.info(f"Return Detect Language Procedure Result")
            return result_list

        except Exception as e:
            logger.warning(f"Detect Language procedure failed: {e}")
            return None


class SvSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        try:
            logger.info("Sv Sentiment Procedure")
            tokens = models.sv_sentiment_tokenizer.encode(_input, return_tensors="pt")
            result = models.sv_sentiment_model(tokens)
            output_np = result.logits[0].detach().numpy()
            output = softmax(output_np)
            logger.info(f"Return Sv Sentiment Procedure Result")
            return output

        except Exception as e:
            logger.warning(f"Sv Sentiment procedure failed: {e}")
            return None


class EnSentimentProcedure(Procedure):
    async def run_procedure(self, _input):
        try:
            logger.info("En Sentiment Procedure")
            tokens = models.en_sentiment_tokenizer.encode(_input, return_tensors="pt")
            result = models.en_sentiment_model(tokens)
            output_np = result.logits[0].detach().numpy()
            output = softmax(output_np)
            logger.info(f"Return En Sentiment Procedure Result")
            return output

        except Exception as e:
            logger.warning(f"En Sentiment procedure failed: {e}")
            return None


class SvSummarizeTextProcedure(Procedure):
    async def run_procedure(self, _input):
        try:
            logger.info("Sv Summarize Procedure")
            tokens = models.sv_summarize_text_tokenizer(_input, return_tensors="pt").input_ids
            outputs = models.sv_summarize_text_model.generate(input_ids=tokens, max_length=130, num_beams=5, num_return_sequences=1)
            result = models.sv_summarize_text_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            logger.info(f"Return Sv Summarize Procedure Result")
            return result[0]

        except Exception as e:
            logger.warning(f"Sv Summarize Procedure failed: {e}")
            return None
