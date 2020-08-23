from fastapi import FastAPI
from app.api.summary import summary


app = FastAPI(openapi_url="/api/v1/summary/openapi.json", docs_url="/api/v1/summary/docs")

app.include_router(summary, prefix='/api/v1/summary', tags=['summary'])