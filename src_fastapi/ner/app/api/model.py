# Importing libraries
from pydantic import BaseModel
from typing import Optional, List


# Model for recieveing input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for NER Service response, first created a model for a single entity and followed by a response for the complete service
class NEREntity(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int


class NERResponse(BaseModel):
    entites: List[NEREntity]
