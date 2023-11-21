import logging

from scipy.special import softmax

from app.api.ai.ai_executions.ABCExecutions import AIExecution
from app.api.ai.ai_utils.ai_model_classes import EnSentimentModel

logger = logging.getLogger(__name__)
model = EnSentimentModel()


class EnSentimentExecution(AIExecution):
    async def run_procedure(self, _input):
        try:
            logger.info("En Sentiment Procedure")
            # lower_case_input = _input.lower().replace(",", "").replace(".", "")
            tokens = model.tokenizer.encode(_input, return_tensors="pt")

            if self.is_tensors_within_limit(tokens, model.max_position_embeddings) is False:
                return None

            result = model.model(tokens)
            output_np = result.logits[0].detach().numpy()
            output = softmax(output_np)
            logger.info(f"Return En Sentiment Procedure Result")
            return output

        except Exception as e:
            logger.warning(f"En Sentiment procedure failed: {e}")
            return None

    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        tensor_size = tokens.size(dim=1)
        if tensor_size > max_position_embeddings:
            logger.warning(f"Token size larger than model max position embeddings: {tensor_size}")
            return False
        return True
