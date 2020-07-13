# Importing packages: streamlit for the frontend, requests to make the api calls
import streamlit as st
import requests
import json


class MakeCalls:
    def __init__(self, url: str) -> None:
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def model_list(self, service: str) -> list:
        """
        Making an API request to backend service to get the details for each service. This function returns, list of names of trained models 
        :param service: NLP service that is being used.
        :return: List of names of trained models
        """
        model_info_url = self.url + f"/v1/{service}/info"
        models = requests.get(url=model_info_url)
        return json.loads(models.text)

    def run_inference(self, service: str, model: str, text: str, query: str = None):
        """
        This function is used to send the api request for the actual service for the specifed model to the
        :param service: String for the actual service.
        :param model: Model that is slected from the drop down.
        :param text: Input text that is used for analysis and to run inference.
        :param query: Input query for Information extraction use case.
        :return: results from the inference done by the model.
        """
        inference_enpoint = self.url + f"/v1/{service}/predict"

        payload = {"model": model.lower(), "text": text.lower(), "query": query.lower()}
        result = requests.post(
            url=inference_enpoint, headers=self.headers, data=json.dumps(payload)
        )
        return json.loads(result.text)


def disaply_page(service: str, models_dict: dict):
    """
    This function is used to generate the page for each service. It returns,
    :param service: String of the service being selected from the side bar.
    :param models: List of trained models that have been trained for this service.
    :return: model, input_text run_button: Selected model from the drop down, input text by the user and run botton to kick off the process.
    """
    st.header(service)
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
    if service == "Information Extraction":
        query: str = st.text_input("Enter query here.")
        run_button: bool = st.button("Run")
        return model, input_text, query, run_button
    else:
        run_button: bool = st.button("Run")
        return model, input_text, run_button


def main():
    st.title("Insight")
    st.sidebar.header("Select the NLP Service")
    service_options = st.sidebar.selectbox(
        label="",
        options=[
            "Project Insight",
            "News Classification",
            "Named Entity Recognition",
            "Sentiment Analysis",
            "Summarization",
            "Information Extraction",
        ],
    )
    service = {
        "Project Insight": "about",
        "News Classification": "classification",
        "Named Entity Recognition": "ner",
        "Sentiment Analysis": "sentiment",
        "Summarization": "summ",
        "Information Extraction": "qna",
    }
    apicall = MakeCalls(url="http://127.0.0.1:8000")
    if service[service_options] == "about":
        st.header("This is the Project Insight about Page...")
    else:
        models_info = apicall.model_list(service=service[service_options])
        if service_options == "Information Extraction":
            model, input_text, query, run_button = disaply_page(service_options, models_info)
        else:
            model, input_text, run_button = disaply_page(service_options, models_info)
            query = str()

        if run_button:
            result = apicall.run_inference(
                service=service[service_options],
                model=model,
                text=input_text,
                query=query,
            )
            st.write(result)


if __name__ == "__main__":
    main()
