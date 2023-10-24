from abc import ABC, abstractmethod


class Procedure(ABC):
    @abstractmethod
    def run_procedure(self, _input):
        pass
