# Importing libraries
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# Model for recieveing input
class Input(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


# Model for classification service response
class ClassResponse(BaseModel):
    category: str
    confidence: float