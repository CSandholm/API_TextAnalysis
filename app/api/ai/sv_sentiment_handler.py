import sv_sentiment_procedure


class SentimentHandler:
    def __init__(self, input):
        self.input = input
        self.sentiment = sv_sentiment_procedure

    async def get_sentiment(self):
        sentiment_result = await self.sentiment.calc_sentiment(self.input)
        return sentiment_result
