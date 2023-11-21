import logging

from app.api.ai.ai_utils.ai_model_classes import EnSvTranslationModel, SvEnTranslationModel

logger = logging.getLogger(__name__)
sv_model = SvEnTranslationModel()
en_model = EnSvTranslationModel()


class TranslationProcedure:
    async def run_procedure(self, _input, src_lang):
        logger.info("Translation Procedure")
        try:
            if src_lang != "sv":
                logger.info("Translating English to Swedish")
                input_ids = en_model.tokenizer(_input, return_tensors="pt").input_ids

                if self.is_tensors_within_limit(input_ids, en_model.max_position_embeddings) is False:
                    return None

                outputs = en_model.model.generate(input_ids=input_ids, num_beams=10, num_return_sequences=1)
                result = en_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translation = result[0]
                logger.info("Return Translation Procedure Result")
                return translation

            else:
                logger.info("Translating Swedish to English")
                input_ids = sv_model.tokenizer(_input, return_tensors="pt").input_ids

                if self.is_tensors_within_limit(input_ids, sv_model.max_position_embeddings) is False:
                    return None

                outputs = sv_model.model.generate(input_ids=input_ids, num_beams=10, num_return_sequences=1)
                result = sv_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                translation = result[0]
                logger.info("Return Translation Procedure Result")
                return translation

        except Exception as e:
            logger.warning(f"Translation procedure failed: {e}")
            return None

    @classmethod
    def is_tensors_within_limit(cls, tokens, max_position_embeddings):
        tensor_size = tokens.size(dim=1)
        if tensor_size > max_position_embeddings:
            logger.warning(f"Token size larger than model max position embeddings: {tensor_size}")
            return False
        return True
