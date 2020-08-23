from fastapi import FastAPI
from app.api.sentiment import sentiment


app = FastAPI(
    openapi_url="/api/v1/sentiment/openapi.json", docs_url="/api/v1/sentiment/docs"
)

app.include_router(sentiment, prefix="/api/v1/sentiment", tags=["sentiment"])
