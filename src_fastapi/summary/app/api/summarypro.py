import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import json

device = torch.device("cpu")


class SummarizerProcessor:
    def __init__(self, model: str = None, service: str = "summ"):
        if model is None:
            model = "t5"

        # path to all the files that will be used for inference
        self.path = f".app/api/{model}"
        self.model_path = self.path + "model.bin"
        self.config_path = self.path + "config.json"

        # Selecting the correct model based on the passed madel input. Default t5
        if model == "t5":
            self.config = T5Config.from_json_file(self.config_path)
            self.model = T5ForConditionalGeneration(self.config)
            self.tokenizer = T5Tokenizer.from_pretrained(self.path)
        else:
            raise Exception("This model is not supported")

        self.model.eval()
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))

        self.text = str()

    def tokenize(self, query: str = None):
        """
        Method to tokenize the textual input
        :param query: Query in case of Question Answering service.
        :return: Returns encoded text for inference
        """
        inputs = self.tokenizer.encode_plus(
            self.text,
            query,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_tensors="pt",
            truncation=True,
        )

        return inputs

    def preprocess(self):
        # Remove quotes and add summarize to the text
        self.text = self.text.replace('"', "")
        self.text = self.text.replace("\n", " ")
        self.text = "summarize: " + self.text
        return self.text

    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input qwuery in case of QnA
        :return: correct category and confidence for that category
        """
        self.text = input_text
        self.text = self.preprocess()
        original_length = len(self.text)
        tokenized_inputs = self.tokenize(query)
        input_ids = tokenized_inputs["input_ids"]
        outputs = self.model.generate(
            input_ids=input_ids, max_length=50, num_beams=4, early_stopping=True,
        )
        preds = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in outputs
        ]
        preds = str(preds)[2:-2]
        summ_length = len(preds)
        return preds, summ_length, original_length
