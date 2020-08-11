# Importing libraries
from pydantic import BaseModel
from typing import Optional
from enum import Enum


# Model for recieveing input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for enumerating over the service types
class ServiceName(str, Enum):
    sentiment = "sentiment"
    classification = "classification"
    ner = "ner"
    summ = "summ"
    qna = "qna"
