import logging
import torch

from app.api.ai.ai_executions.ABCExecutions import AIExecution
from app.api.ai.ai_utils import languages
from app.api.ai.ai_utils.ai_model_classes import DetectLanguageModel

logger = logging.getLogger(__name__)

model = DetectLanguageModel()


class DetectLanguageExecution(AIExecution):
    async def run_procedure(self, _input):
        try:
            logger.info("Detect Language Procedure")
            tokens = model.tokenizer.encode(_input, return_tensors='pt')

            if self.is_tensors_within_limit(tokens, model.max_position_embeddings) is False:
                raise Exception("Ajabajja")

            result = model.model(tokens)
            result_logits = result.logits
            result_index = int(torch.argmax(result_logits))
            result_list = languages.get_language_code(result_index)
            score = str(round(result_logits[0][result_index].item()))
            result_list.append(score)
            logger.info(f"Return Detect Language Procedure Result")
            return result_list

        except Exception as e:
            logger.warning(f"Detect Language procedure failed: {e}")
            return e

    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        tensor_size = tokens.size(dim=1)
        if tensor_size > max_position_embeddings:
            logger.warning(f"Token size larger than model max position embeddings: {tensor_size}")
            return False
        return True
