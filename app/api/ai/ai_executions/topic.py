import logging

from app.api.ai.ai_executions.ABCExecutions import AIExecution
from app.api.ai.ai_utils.ai_model_classes import SvTopicModel

logger = logging.getLogger(__name__)
topic = SvTopicModel()


class TopicExecution(AIExecution):
    async def run_procedure(self, _input):
        try:
            predictions = topic.model.extract_keywords(_input)

            keyword_list = []
            for item in predictions:
                keyword_list.append(item[0])

            return keyword_list

        except Exception as e:
            raise Exception(f"Could not execute Topic execution : {e}")

    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        pass
