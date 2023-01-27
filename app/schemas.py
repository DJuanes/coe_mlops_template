from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Text(BaseModel):
    text: str = Query(None, min_length=1)


class PredictPayload(BaseModel):
    texts: List[Text]

    @validator("texts")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("La lista de textos a clasificar no puede estar vacía.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    {"text": "Transfer learning with transformers for text classification."},
                    {"text": "Generative adversarial networks in both PyTorch and TensorFlow."},
                ]
            }
        }
