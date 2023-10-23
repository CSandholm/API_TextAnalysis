from pydantic import BaseModel


class Response(BaseModel):
    input: str
