import asyncio
import re

from app.api.ai.ai_executions.en_sentiment import EnSentimentExecution
from app.api.ai.ai_executions.translation import TranslationProcedure


class MulSentimentHandler:
    def __init__(self, _input):
        self.input = _input
        self.src_lang = "any"
        self.sentiment = EnSentimentExecution()
        self.translator = TranslationProcedure()

    async def get_sentiment(self):
        try:
            clean_input = re.sub(r'[^A-Za-z0-9äöüÄÖÜß ]', '', self.input).lower()
            task = asyncio.create_task(self.translator.run_procedure(clean_input, self.src_lang))
            translation = await task

            task = asyncio.create_task(self.sentiment.run_procedure(translation))
            sentiment_result = await task
            return sentiment_result
        except Exception as e:
            raise Exception(f"{e}")
