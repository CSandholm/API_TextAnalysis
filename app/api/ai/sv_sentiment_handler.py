import asyncio
import json

from app.api.ai.procedures import SvSentimentProcedure


class SvSentimentHandler:
    def __init__(self, _input):
        self.input = _input
        self.sentiment = SvSentimentProcedure()

    async def get_sentiment(self):
        task = asyncio.create_task(self.sentiment.run_procedure(self.input))
        sentiment_result = await task
        return sentiment_result
