# Importing Libraries for the server side script
from fastapi import FastAPI
import json

# Importing the models for standardizing the inputs to the service
from model import Input, ServiceName

# Importing various NLP processors to the app
from sentimentpro import SentimentProcessor
from classificationpro import ClassProcessor
from summarypro import SummarizerProcessor
from nerpro import NerProcessor


# Declaring the App
app = FastAPI()


# Root path
@app.get("/")
async def root():
    """
    Call to the Index/Root path of the API service
    :return: A simple key value pair to the calling service
    """
    return {"message": "Hello from Insight"}


# Path to provide details of the services loaded in the backend. Details are obtained from the config json file
@app.get("/v1/{service}/info")
async def get_models(service: ServiceName):
    """
    This method returns model details to the front end. Based on the service argument
    * :param service: Service can from one of the services such as: Classification, sentiment analysis etc.
    * :return:
    """
    with open("config.json") as f:
        config = json.load(f)
    model_info = config[service]
    return model_info


# Path for classification service
@app.post("/v1/classification/predict")
async def classification(item: Input):
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


# Path for named entity recognition service
@app.post("/v1/ner/predict")
async def named_entity_recognition(item: Input):
    """
    This function is used to perform Named Entity Recognition for the inpur text
    * :param item: This is the payload that is sent to the server. The structure of item defined above
    * :return: Returns a list of dictionary. It will be of the structure:
    [
        {
            text: entity_1,
            type: entity_type_1,
            start: start_char,
            end: end_char
        },
        {
            text: enttity_2,
            type: entity_type_2,
            start: start_char,
            end: end_char
        }
    ]
    """
    ner_process = NerProcessor(model=item.model.lower())
    text = item.text
    result = ner_process.inference(input_text=text)
    return {"entites": result}


# Path for sentiment analysis service
@app.post("/v1/sentiment/predict")
async def sentiment(item: Input):
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


# Path for Summarization service
@app.post("/v1/summ/predict")
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
    output_dict["summary length"] = summary_length
    output_dict["original length"] = original_length
    return output_dict


# Path for qna service
@app.post("/v1/qna/predict")
async def question_answering(item: Input):
    pass
