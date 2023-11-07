import asyncio

from app.api.ai.ai_executions.translation import TranslationProcedure


class TranslationHandler:
    def __init__(self, _input, src_lang, tgt_lang):
        self.input = _input
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator = TranslationProcedure()

    async def get_translation(self):
        task = asyncio.create_task(self.translator.run_procedure(self.input, self.src_lang))
        result = await task
        return result
