import asyncio

from app.api.ai.ai_executions.en_sentiment import EnSentimentExecution


class EnSentimentHandler:
    def __init__(self, _input):
        self.input = _input
        self.sentiment = EnSentimentExecution()

    async def get_sentiment(self):
        task = asyncio.create_task(self.sentiment.run_procedure(self.input))
        sentiment_result = await task
        return sentiment_result
