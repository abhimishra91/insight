# Importing packages: streamlit for the frontend, requests to make the api calls
import streamlit as st
import requests
import json


class MakeCalls:
    def __init__(self, url: str = "http://localhost:8080/") -> None:
        """
        Constructor for the MakeCalls class. This class is used to perform API calls to the backend service.
        :param url: URL of the server. Default value is set to local host: http://localhost:8080
        """
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def model_list(self, service: str) -> dict:
        """
        Making an API request to backend service to get the details for each service. This function returns, list of names of trained models 
        :param service: NLP service that is being used.
        :return: List of names of trained models
        """
        model_info_url = self.url + f"api/v1/{service}/info"
        models = requests.get(url=model_info_url)
        return json.loads(models.text)

    def run_inference(
        self, service: str, model: str, text: str, query: str = None
    ) -> json:
        """
        This function is used to send the api request for the actual service for the specifed model to the
        :param service: String for the actual service.
        :param model: Model that is slected from the drop down.
        :param text: Input text that is used for analysis and to run inference.
        :param query: Input query for Information extraction use case.
        :return: results from the inference done by the model.
        """
        inference_enpoint = self.url + f"api/v1/{service}/predict"

        payload = {"model": model.lower(), "text": text, "query": query.lower()}
        result = requests.post(
            url=inference_enpoint, headers=self.headers, data=json.dumps(payload)
        )
        return json.loads(result.text)


class Display:
    def __init__(self):
        st.title("Insight")
        st.sidebar.header("Select the NLP Service")
        self.service_options = st.sidebar.selectbox(
            label="",
            options=[
                "Project Insight",
                "News Classification",
                "Named Entity Recognition",
                "Sentiment Analysis",
                "Summarization",
            ],
        )
        self.service = {
            "Project Insight": "about",
            "News Classification": "classification",
            "Named Entity Recognition": "ner",
            "Sentiment Analysis": "sentiment",
            "Summarization": "summary",
        }

    def static_elements(self):
        return self.service[self.service_options]

    def dynamic_element(self, models_dict: dict):
        """
        This function is used to generate the page for each service.
        :param service: String of the service being selected from the side bar.
        :param models_dict: Dictionary of Model and its information. This is used to render elements of the page.
        :return: model, input_text run_button: Selected model from the drop down, input text by the user and run botton to kick off the process.
        """
        st.header(self.service_options)
        model_name = list()
        model_info = list()
        for i in models_dict.keys():
            model_name.append(models_dict[i]["name"])
            model_info.append(models_dict[i]["info"])
        st.sidebar.header("Model Information")
        for i in range(len(model_name)):
            st.sidebar.subheader(model_name[i])
            st.sidebar.info(model_info[i])
        model: str = st.selectbox("Select the Trained Model", model_name)
        input_text: str = st.text_area("Enter Text here")
        if self.service == "qna":
            query: str = st.text_input("Enter query here.")
        else:
            query: str = "None"
        run_button: bool = st.button("Run")
        return model, input_text, query, run_button


def main():

    page = Display()
    service = page.static_elements()
    apicall = MakeCalls()
    if service == "about":
        st.header("NLP as a Service")
        st.write(
            "The users can leverage fine-tuned language models to perform multiple downstream tasks, via GUI and API access."
        )
        st.write(
            "Insight backed in designed in a way that developers can also add-in their own fine-tuned models on different datasets and use them for prediction."
        )
        st.write(
            "To use this solution, select a service from the dropdown in the side bar. Details of pre-loaded  pre-trained model will be available based on the service."
        )
        st.write(
            "Fill in the text on which you want to run the service and then let the magic happen."
        )
    else:
        model_details = apicall.model_list(service=service)
        model, input_text, query, run_button = page.dynamic_element(model_details)
        if run_button:
            with st.spinner(text="Getting Results.."):
                result = apicall.run_inference(
                    service=service,
                    model=model.lower(),
                    text=input_text,
                    query=query.lower(),
                )
            st.write(result)


if __name__ == "__main__":
    main()
