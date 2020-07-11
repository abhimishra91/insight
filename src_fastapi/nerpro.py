from logging import disable
import spacy


class NerProcessor:
    def __init__(self, model: str = None, service: str = "ner"):
        if model is None:
            model = "spacy"

        # path to all the files that will be used for inference
        self.path = f"./{service}/{model}/"
        self.model_path = self.path + "model.bin"
        self.config_path = self.path + "config.json"

        # Selecting the correct model based on the passed madel input. Default t5
        if model == "spacy":
            self.model = spacy.load(self.path, disable=["tagger", "parser"])
        else:
            raise Exception("This model is not supported")

    def tokenize(self):
        pass

    def preprocess(self):
        self.text = self.text.replace('"', "")
        self.text = self.text.replace("\n", " ")
        return self.text

    def run_spacy_inference(self):
        result = dict()
        result_list = list()
        docs = self.model.pipe(self.text, disable=["tagger", "parser"])
        for ent in docs.ents:
            result = {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            result_list.append(result)
        return result_list

    def inference(self, input_text: str, query: str = None):
        """
        Method to perform the inference
        :param input_text: Input text for the inference
        :param query: Input query in case of QnA
        :return:A list of dictionary 1 dict for each entity.
        """
        self.text = input_text
        self.text = self.preprocess()
        if self.model == "spacy":
            results = self.run_spacy_inference()
        else:
            raise Exception("This model is not supported")
        return results
