from pydantic import BaseModel


class Request(BaseModel):
    input: str


class TranslationRequest(BaseModel):
    input: str
    src_lang: str
    tgt_lang: str
