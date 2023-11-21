import logging

from app.api.ai.ai_executions.ABCExecutions import AIExecution
from app.api.ai.ai_utils.ai_model_classes import SvSummarizeTextModel

logger = logging.getLogger(__name__)
model = SvSummarizeTextModel()


class SvSummarizeTextExecution(AIExecution):
    async def run_procedure(self, _input):
        try:
            logger.info("Sv Summarize Procedure")
            tokens = model.tokenizer(_input, return_tensors="pt").input_ids

            if self.is_tensors_within_limit(tokens, model.max_position_embeddings) is False:
                return None

            outputs = model.model.generate(input_ids=tokens, max_length=130, num_beams=5)
            result = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            logger.info(f"Return Sv Summarize Procedure Result")

            return result[0]

        except Exception as e:
            logger.warning(f"Sv Summarize Procedure failed: {e}")
            return None

    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        tensor_size = tokens.size(dim=1)
        if tensor_size > max_position_embeddings:
            logger.warning(f"Token size larger than model max position embeddings: {tensor_size}")
            return False
        return True
