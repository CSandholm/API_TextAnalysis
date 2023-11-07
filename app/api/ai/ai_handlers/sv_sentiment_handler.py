import asyncio

from app.api.ai.ai_executions.sv_sentiment import SvSentimentExecution


class SvSentimentHandler:
    def __init__(self, _input):
        self.input = _input
        self.sentiment = SvSentimentExecution()

    async def get_sentiment(self):
        task = asyncio.create_task(self.sentiment.run_procedure(self.input))
        sentiment_result = await task
        return sentiment_result
