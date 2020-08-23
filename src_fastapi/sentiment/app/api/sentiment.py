import json
from typing import List
from fastapi import APIRouter

from app.api.model import Input, SentimentResponse
from app.api.sentimentpro import SentimentProcessor

sentiment = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@sentiment.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: sentiment, sentiment analysis etc.
    * :return:
    """
    with open("./app/api/config.json") as f:
        config = json.load(f)
    return config

# Path for sentiment analysis service
@sentiment.post("/predict", response_model=SentimentResponse)
async def senti(item: Input):
    """
    This function will return the sentiment of the input text. Positive, negative along with the confidence for the sentiment.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: A dictionary for each input with sentiment and the confidence for the prediction.
    {
        "sentiment": "positive/negative/neutral",
        "confidence": confidence
    }
    """
    output_dict = dict()
    sentiment_process = SentimentProcessor(model=item.model.lower())
    text = item.text
    perdiction, confidence = sentiment_process.inference(input_text=text)
    output_dict["sentiment"] = perdiction
    output_dict["confidence"] = confidence
    return output_dict