# Importing Libraries for the server side script
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json

from sentimentpro import SentimentProcessor
from classificationpro import ClassProcessor
from summarypro import SummarizerProcessor


class Item(BaseModel):
    model: str
    text: str
    query: Optional[str] = None


app = FastAPI()


@app.get("/")
async def root():
    """
    Call to the Index/Root path of the API service
    :return: A simple key value pair to the calling service
    """
    return {"message": "Hello from Insight"}


@app.get("/v1/{service}/info")
async def get_models(service: str):
    with open("config.json") as f:
        config = json.load(f)
    model_list = config[service]
    return model_list


@app.post("/v1/classification/predict")
async def classification(item: Item):
    """
    This is the API method for classification based models and related task.
    :param item: This is the payload that is sent to the server. The structure of item defined above
    :return: Label/category of the Input text and the confidence for the prediction.
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


@app.post("/v1/ner/predict")
async def named_entity_recognition(item: Item):
    """
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
    """
    This function will return the sentiment of the input text. Positive, negative along with the confidence for the sentiment.
    :param item: This is the payload that is sent to the server. The structure of item defined above
    :return: A dictionary for each input with sentiment and the confidence for the prediction.
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


@app.post("/v1/summ/predict")
async def summarization(item: Item):
    """
        This function will return the sentiment of the input text. Positive, negative along with the confidence for the sentiment.
        :param item: This is the payload that is sent to the server. The structure of item defined above
        :return: A dictionary for each input with summary for the input text lenght of the new summary and lenght of the original input.
        {
            "summary": "Multiline summary",
            "length": len,
            "original text length": len
        }
    """
    output_dict = dict()
    summary_process = SummarizerProcessor(model=item.model.lower())
    text = item.text
    summary, summary_length, original_length = summary_process.inference(input_text=text)
    output_dict["summary"] = summary
    output_dict["summary length"] = summary_length
    output_dict["original length"] = original_length
    return output_dict


@app.post("/v1/qna/predict")
async def question_answering(item: Item):
    pass
