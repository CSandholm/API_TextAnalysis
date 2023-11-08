import logging

logger = logging.getLogger(__name__)


class TopicExecution:
    async def run_procedure(self, _input, vocabulary, stopwords):
        try:
            logger.info("Topic Procedure")
            filtered_words = await self.filter_input(_input, stopwords)
            mapped_tokens = []
            logger.info("Synonyms to lower")
            for key, synonyms in vocabulary.items():
                for i, synonym in enumerate(synonyms):
                    synonyms[i] = synonym.lower()

            for word in filtered_words:
                key = vocabulary.get(word)
                if key and key not in mapped_tokens:
                    mapped_tokens.append(key)
                else:
                    for key, synonyms in vocabulary.items():
                        if word in synonyms and key not in mapped_tokens:
                            mapped_tokens.append(key)
                            break

            return mapped_tokens
        except Exception:
            raise Exception("Could not execute Topic execution")

    @classmethod
    async def filter_input(cls, _input, stopwords):
        clean_input = _input.lower().split()
        filtered_words = []
        for word in stopwords:
            word = word.lower()
        for word in clean_input:
            if word not in stopwords:
                filtered_words.append(word)
        return filtered_words
