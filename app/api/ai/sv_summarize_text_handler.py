import asyncio

from app.api.ai.procedures import SvSummarizeTextProcedure


class SvSummarizeTextHandler:
    def __init__(self, _input):
        self.input = _input
        self.summarizer = SvSummarizeTextProcedure()

    async def get_summary(self):
        task = asyncio.create_task(self.summarizer.run_procedure(self.input))
        summarize_result = await task
        return summarize_result
