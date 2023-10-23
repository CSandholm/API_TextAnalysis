import sentiment_procedure


class SentimentHandler:
    def __init__(self):
        self.sentiment = sentiment_procedure

    async def get_sentiment(self, input):
        sentiment = await self.sentiment.calc_sentiment(input)
        return sentiment
