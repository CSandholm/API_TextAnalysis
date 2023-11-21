import asyncio

from app.api.ai.ai_executions.sv_summarize_text import SvSummarizeTextExecution


class SvSummarizeTextHandler:
    def __init__(self, _input):
        self.input = _input
        self.summarizer = SvSummarizeTextExecution()

    async def get_summary(self):
        task = asyncio.create_task(self.summarizer.run_procedure(self.input))
        summarize_result = await task
        return summarize_result
