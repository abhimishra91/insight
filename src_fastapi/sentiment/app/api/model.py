# Importing libraries
from pydantic import BaseModel
from typing import Optional


# Model for recieveing input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for sentiment analysis service response
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float