# Get topic vocabulary from analytics?

import asyncio

from app.api.ai.procedures import TopicProcedure


class TopicHandler:
    def __init__(self, _input):
        self.input = _input
        self.topic_procedure = TopicProcedure()

    async def get_topics(self):
        task = asyncio.create_task(self.topic_procedure.run_procedure(self.input))
        topic_result = await task
        return topic_result
