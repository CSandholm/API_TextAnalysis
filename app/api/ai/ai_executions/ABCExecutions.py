from abc import ABC, abstractmethod


class AIExecution(ABC):
    @abstractmethod
    def run_procedure(self, _input):
        pass

    @abstractmethod
    def is_tensors_within_limit(self, tokens, max_position_embeddings):
        pass
