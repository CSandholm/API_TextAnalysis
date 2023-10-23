from ABCProcedures import Procedure


class SvSentimentProcedure(Procedure):
    async def run_procedure(self, path, path2):
        tokens = await self.AutoTokenizer.encode(input, return_tensors="pt")
        result = await self.AutoModelForSequenceClassification(tokens)
        output_np = result.logits[0].detach().cpu().numpy()
        output = self.softmax(output_np)
        return output
