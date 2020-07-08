import torch
from transformers import DistilBertTokenizerFast
import json
from classification.distilbert.network import DistillBERTClass

device = torch.device("cpu")


class ClassProcessor:
    def __init__(self, model: str = None, service: str = "classification"):
        if model is None:
            model = "distilbert"

        self.path = f"./{service}/{model}/"
        self.mapping = self.path + "mapping.json"
        self.model_path = self.path + "model.bin"
        if model == "distilbert":
            self.model = DistillBERTClass()
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.path)
        else:
            self.model = DistillBERTClass()
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.path)

        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

        with open(self.mapping) as f:
            self.config = json.load(f)

    def tokenize(self, input_text: str, query: str = None):
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

    def lookup(self):
        return self.config[str(int(self.pred.item()))]

    def inference(self, input_text: str, query: str = None):
        self.tokenized_inputs = self.tokenize(input_text, query)
        self.input_ids = self.tokenized_inputs["input_ids"]
        self.attention_mask = self.tokenized_inputs["attention_mask"]
        self.outputs = self.model(input_ids=self.input_ids, attention_mask=self.attention_mask)
        self._, self.pred = torch.max(self.outputs, dim=1)
        sentiment_class = self.lookup()
        self.conf, self.pos = torch.max(torch.nn.functional.softmax(self.outputs, dim=1), dim=1)
        return sentiment_class, self.conf.item()