# Importing libraries
from pydantic import BaseModel
from typing import Optional


# Model for recieveing input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for summarization service response
class SummaryResponse(BaseModel):
    summary: str
    summary_length: int
    original_length: int