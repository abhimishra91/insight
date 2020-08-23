from fastapi import FastAPI
from app.api.classification import classification


app = FastAPI(openapi_url="/api/v1/classification/openapi.json", docs_url="/api/v1/classification/docs")

app.include_router(classification, prefix='/api/v1/classification', tags=['classification'])    