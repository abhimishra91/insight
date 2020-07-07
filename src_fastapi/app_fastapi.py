# Importing Libraries for the server side script
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn


class Item(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


app = FastAPI()


@app.get("/")
async def root():
    r"""
    Call to the Index/Root path of the API service
    :return: A simple key value pair to the calling service
    """
    return {"message": "Hello from Insight"}


@app.post("/v1/classification/predict")
async def classification(item: Item):
    r"""
    This is the API method for classification based models and related task.
    :param item: This is the payload that is sent to the server. The structure of item defined above
    :return: Label/category of the Input text and the confidence for the prediction.
    {
        "category":"category_1",
        "confidence":confidence
    }
    """
    pass


@app.post("/v1/ner/predict")
async def ner(item: Item):
    r"""
    This function is used to perform Named Entity Recognition for the inpur text
    :param item: This is the payload that is sent to the server. The structure of item defined above
    :return: Returns a list of dictionary. It will be of the structure:
    [
        {
            text: entity_1,
            type: entity_type_1,
            confidence: confidence_1
        },
        {
            text: enttity_2,
            type: entity_type_2,
            confidence: confidence_2
        }
    ]
    """
    pass


@app.post("/v1/sentiment/predict")
async def sentiment(item: Item):
    r"""
    This function will return the sentiment of the input text. Positive, negative along with the confidence for the sentiment.
    :param item: This is the payload that is sent to the server. The structure of item defined above
    :return: A dictionary for each input with sentiment and the confidence for the prediction.
    {
        "sentiment": "positive/negative/neutral",
        "confidence": confidence
    }
    """
    pass


@app.post("v1/summ/predict")
async def summ(item: Item):
    r"""
        This function will return the sentiment of the input text. Positive, negative along with the confidence for the sentiment.
        :param item: This is the payload that is sent to the server. The structure of item defined above
        :return: A dictionary for each input with summary for the input text lenght of the new summary and lenght of the original input.
        {
            "summary": "Multiline summary",
            "lenght": len,
            "original text lenght": len
        }
        """
    pass


@app.post("v1/qna/predict")
async def qna(item: Item):
    pass
