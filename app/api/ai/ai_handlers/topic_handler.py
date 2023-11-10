import asyncio

from app.api.ai.ai_executions.topic import TopicExecution


class TopicHandler:
    def __init__(self, _input):
        self.input = _input
        self.topic_procedure = TopicExecution()

    async def get_topics(self):
        task = asyncio.create_task(self.topic_procedure.run_procedure(self.input))
        topic_result = await task
        return topic_result
