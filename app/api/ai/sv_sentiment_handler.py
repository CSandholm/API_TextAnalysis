from sv_sentiment_procedure import SvSentiment


class SentimentHandler:
    def __init__(self, input):
        self.input = input
        self.sentiment = SvSentiment()

    async def get_sentiment(self):
        sentiment_result = await self.sentiment.calc_sentiment(self.input)
        return sentiment_result
