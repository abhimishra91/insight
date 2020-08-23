import json
from typing import List
from fastapi import APIRouter

from app.api.model import Input, SummaryResponse
from app.api.summarypro import SummarizerProcessor

summary = APIRouter()

# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@summary.get("/info")
async def get_models():
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: sentiment, sentiment analysis etc.
    * :return:
    """
    with open("./app/api/config.json") as f:
        config = json.load(f)
    return config


# Path for Summarization service
@summary.post("/predict", response_model=SummaryResponse)
async def summarization(item: Input):
    """
    This function will return the summary of the input text. Will be generated only for text greater than 150 words.
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: A dictionary for each input with summary for the input text lenght of the new summary and lenght of the original input.
        {
            "summary": "Multiline summary",
            "length": len,
            "original text length": len
        }
    """
    output_dict = dict()
    text = item.text
    if len(text) < 150:
        summary = text
        summary_length = len(text)
        original_length = len(text)
    else:
        summary_process = SummarizerProcessor(model=item.model.lower())
        summary, summary_length, original_length = summary_process.inference(
            input_text=text
        )
    output_dict["summary"] = summary
    output_dict["summary_length"] = summary_length
    output_dict["original_length"] = original_length
    return output_dict
