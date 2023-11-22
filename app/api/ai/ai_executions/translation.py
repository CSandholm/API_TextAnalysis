import logging

from app.api.ai.ai_utils.ai_model_classes import EnSvTranslationModel, SvEnTranslationModel, MulEnTranslationModel

logger = logging.getLogger(__name__)
sv_model = SvEnTranslationModel()
en_model = EnSvTranslationModel()
mul_model = MulEnTranslationModel()


class TranslationProcedure:
    async def run_procedure(self, _input, src_lang):
        logger.info("Translation Procedure")
        try:
            if src_lang == "any" or src_lang == "mul":
                translation = self.translate(mul_model, _input)
                logger.info(f"Translation: {translation}")
                return translation
            if src_lang == "en":
                return self.translate(en_model, _input)
            else:
                return self.translate(sv_model, _input)

        except Exception as e:
            logger.warning(f"Translation procedure failed: {e}")
            return None

    def translate(self, model, _input):
        logger.info("Translating")
        input_ids = model.tokenizer(_input, return_tensors="pt").input_ids

        if self.is_tensors_within_limit(input_ids, model.max_position_embeddings) is False:
            return None

        outputs = model.model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1)
        result = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translation = result[0]
        logger.info("Return Translation Procedure Result")

        return translation

    @classmethod
    def is_tensors_within_limit(cls, tokens, max_position_embeddings):
        tensor_size = tokens.size(dim=1)
        if tensor_size > max_position_embeddings:
            logger.warning(f"Token size larger than model max position embeddings: {tensor_size}")
            return False
        return True
