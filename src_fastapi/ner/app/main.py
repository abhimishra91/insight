from fastapi import FastAPI
from app.api.ner import ner


app = FastAPI(openapi_url="/api/v1/ner/openapi.json", docs_url="/api/v1/ner/docs")

app.include_router(ner, prefix="/api/v1/ner", tags=["ner"])
