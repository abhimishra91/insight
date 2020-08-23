import json
from typing import List
from fastapi import APIRouter

from app.api.model import Input, NEREntity, NERResponse
from app.api.nerpro import NerProcessor

ner = APIRouter()


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@ner.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: ner, sentiment analysis etc.
    * :return:
    """
    with open("./app/api/config.json") as f:
        config = json.load(f)
    return config

# Path for named entity recognition service
@ner.post("/predict", response_model=NERResponse)
async def named_entity_recognition(item: Input):
    """
    This function is used to perform Named Entity Recognition for the inpur text
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: Returns a list of dictionary. It will be of the structure:
    [
        {
            text: entity_1,
            entity_type: entity_type_1,
            start: start_char,
            end: end_char
        },
        {
            text: enttity_2,
            entity_type: entity_type_2,
            start: start_char,
            end: end_char
        }
    ]
    """
    ner_process = NerProcessor(model=item.model.lower())
    text = item.text
    result = ner_process.inference(input_text=text)
    return {"entites": result}
