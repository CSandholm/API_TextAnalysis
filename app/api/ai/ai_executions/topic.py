import logging

from app.api.ai.ai_executions.ABCExecutions import AIExecution
from app.api.ai.ai_utils.ai_model_classes import SvTopicModel

logger = logging.getLogger(__name__)
topic = SvTopicModel()


class TopicExecution(AIExecution):
    async def run_procedure(self, _input):
        try:
            logger.info("Topic Procedure")
            predictions = topic.model.extract_keywords(_input)
            logger.info("Listing keywords")
            keyword_list = []
            for item in predictions:
                keyword_list.append(item[0])

            logger.info("Returning keyword list")
            return keyword_list

        except Exception as e:
            logger.error(f"{e}")
            raise Exception(f"Could not execute Topic execution : {e}")

    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        pass
