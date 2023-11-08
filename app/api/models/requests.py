from typing import Dict, List

from pydantic import BaseModel


class Request(BaseModel):
    input: str


class TopicRequest(BaseModel):
    input: str
    vocabulary: Dict[str, List[str]]
    stopwords: List[str]


class TranslationRequest(BaseModel):
    input: str
    src_lang: str
    tgt_lang: str
