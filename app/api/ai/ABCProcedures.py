import json

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

class Procedure(ABC):
    def __init__(self):
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForSequenceClassification = AutoModelForSequenceClassification
        self.softmax = softmax
    @abstractmethod
    def run_procedure(self, model_path, tokenizer_path):
        model = AutoModelForSequenceClassification(model_path) # This to actual class
        tokenizer = self.AutoTokenizer(tokenizer_path)
        pass
