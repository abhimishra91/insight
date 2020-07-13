import torch
from transformers import RobertaTokenizerFast, DistilBertTokenizerFast
import json
from sentiment.distilbert.network import DistilBertClass
from sentiment.roberta.network import RobertaClass

device = torch.device("cpu")


class SentimentProcessor:
    def __init__(self, model: str = None, service: str = "sentiment"):
        """
        Constructor to the class that does the Sentiment Analysis Processing in the back end
        :param model: Transfomer model that will be used for sentiment analysis
        :param service: string to represent the service, this will be defaulted to sentiment
        """
        if model is None:
            model = "distilbert"
        # path to all the files that will be used for inference
        self.path = f"./{service}/{model}/"
        # json file for mapping of network output to the correct category
        self.mapping = self.path + "mapping.json"
        self.model_path = self.path + "model.bin"
        # Selecting the correct model based on the passed madel input. Default distilbert
        if model == "roberta":
            self.model = RobertaClass()
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.path)
        elif model == "distilbert":
            self.model = DistilBertClass()
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.path)
        else:
            self.model = DistilBertClass()
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.path)

        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

        with open(self.mapping) as f:
            self.config = json.load(f)

    def tokenize(self, input_text: str, query: str = None):
        """
        Method to tokenize the textual input
        :param input_text: Input text
        :param query: Query in case of Question Answering service.
        :return: Returns encoded text for inference
        """
        inputs = self.tokenizer.encode_plus(
            input_text,
            query,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
        )

        return inputs

    def lookup(self, pred):
        """
        Function to perform look up against the mapping json file. Only applicable for classificaiton and sentiment analysis.
        :return: Correct category for the prediction.
        """
        return self.config[str(int(pred.item()))]

    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input qwuery in case of QnA
        :return: correct category and confidence for that category
        """
        tokenized_inputs = self.tokenize(input_text, query)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        _, pred = torch.max(outputs, dim=1)
        sentiment_class = self.lookup(pred)
        conf, pos = torch.max(
            torch.nn.functional.softmax(outputs, dim=1), dim=1
        )
        return sentiment_class, conf.item()
