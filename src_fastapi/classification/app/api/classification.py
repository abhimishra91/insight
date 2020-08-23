import json
from typing import List
from fastapi import APIRouter, HTTPException

from app.api.model import Input, ClassResponse
from app.api.classificationpro import ClassProcessor

classification = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@classification.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: Classification, sentiment analysis etc.
    * :return:
    """
    with open("./app/api/config.json") as f:
        config = json.load(f)
    return config


# Path for classification service
@classification.post("/predict", response_model=ClassResponse)
async def classifiy(item: Input):
    """
    This is the API method for classification based models and related task.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: Label/category of the Input text and the confidence for the prediction.
    {
        "category":"category_1",
        "confidence":confidence
    }
    """
    output_dict = dict()
    class_process = ClassProcessor(model=item.model.lower())
    text = item.text
    perdiction, confidence = class_process.inference(input_text=text)
    output_dict["category"] = perdiction
    output_dict["confidence"] = confidence
    return output_dict
