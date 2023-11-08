import asyncio

from app.api.ai.ai_executions.topic import TopicExecution


class TopicHandler:
    def __init__(self, _input, vocabulary, stopwords):
        self.input = _input
        self.topic_procedure = TopicExecution()
        self.vocabulary = vocabulary
        self. stopwords = stopwords

    async def get_topics(self):
        task = asyncio.create_task(self.topic_procedure.run_procedure(self.input, self.vocabulary, self.stopwords))
        topic_result = await task
        return topic_result
