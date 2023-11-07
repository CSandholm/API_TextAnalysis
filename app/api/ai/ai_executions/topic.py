import logging
import json
import re

logger = logging.getLogger(__name__)


class TopicExecution:
    def __init__(self):
        self.vocabulary = self.get_vocabulary()

    async def run_procedure(self, _input):

        tokens = re.findall(r'\b\w+\b', _input.lower())
        mapped_tokens = []

        self.get_vocabulary()

        for token in tokens:
            for common_word, synonyms in self.vocabulary.items():
                if common_word not in mapped_tokens and token in synonyms:
                    mapped_tokens.append(common_word)
                    logger.info(f"Added: {common_word}")
                    break

        return mapped_tokens

    @classmethod
    def get_vocabulary(cls):
        with open("app/api/ai/ai_utils/topic_vocabulary.json", 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        return vocabulary
