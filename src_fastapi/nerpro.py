import spacy


class NerProcessor:
    def __init__(self, model: str = "spacy", service: str = "ner"):
        """
        Constructor to the class that does the Named Entity Recognition in the back end
        :param model: Transfomer model that will be used for Named Entity Recognition
        :param service: string to represent the service, this will be defaulted to "ner"
        """
        self.text = str()
        self.model_name = model

        # path to all the files that will be used for inference
        self.path = f"./{service}/{model}/"
        self.model_path = self.path + "model.bin"
        self.config_path = self.path + "config.json"

        # Selecting the correct model based on the passed madel input. Default t5
        if self.model_name == "spacy":
            self.model = spacy.load("./ner/spacy/", disable=["tagger", "parser"])
        else:
            raise Exception("This model is not supported")

    def tokenize(self):
        pass

    def preprocess(self):
        """
        Method to preprocess the text for T5 model
        :return: self.text
        """
        self.text = self.text.replace('"', "")
        self.text = self.text.replace("\n", " ")
        return self.text

    def run_spacy_inference(self):
        """
        This method has been defined to perform inference for Spcy model
        :return: list of dict
        """
        result = dict()
        result_list = list()
        docs = self.model(self.text, disable=["tagger", "parser"])
        for ent in docs.ents:
            result = {
                "text": ent.text,
                "entity_type": ent.label_,
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
        if self.model_name == "spacy":
            results = self.run_spacy_inference()
        else:
            raise Exception("This model is not supported")
        return results
