# Importing libraries
from pydantic import BaseModel
from typing import Optional, List
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

# Model for classification service response
class ClassResponse(BaseModel):
    category: str
    confidence: float

# Model for sentiment analysis service response
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

# Model for summarization service response
class SummaryResponse(BaseModel):
    summary: str
    summary_length: int
    original_length: int

# Model for NER Service response, first created a model for a single entity and followed by a response for the complete service
class NEREntity(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int

class NERResponse(BaseModel):
    entitxes: List[NEREntity]