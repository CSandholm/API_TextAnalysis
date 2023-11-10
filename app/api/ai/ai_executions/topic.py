import logging

logger = logging.getLogger(__name__)


class TopicExecution:
    async def run_procedure(self, _input):
        try:
            return ["This will be availblalelelel soon"]
        except Exception:
            raise Exception("Could not execute Topic execution")
