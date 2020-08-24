<h1 align="center">Project Insight</h1>

<h2 align="center">NLP as a Service</h2>

<p align="center">
<img alt="Project Insight" src="meta/Insight.png">
</p>

<p align="center">
<a href="https://github.com/abhimishra91/insight/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/abhimishra91/insight"></a>
<a href="https://github.com/abhimishra91/insight/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/abhimishra91/insight"></a>
<a href="https://github.com/abhimishra91/insight/stargazers"><img alt="Github Stars" src="https://img.shields.io/github/stars/abhimishra91/insight"></a>
<a href="https://github.com/abhimishra91/insight/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/abhimishra91/insight"></a>
<a href="https://github.com/abhimishra91/insight/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Contents

1. [Introduction](#section01)
    - [Features](#section01a)
2. [Installation](#section02)
    - [Running the Service](#section02a)
3. [Project Details](#section03)
    - [Demonstration](#section03a)
    - [Directory Details](#section03b)
    - [How to Add a new Model](#section03c)
4. [License](#section04)


<a id='section01'></a>

## Introduction

Project Insight is designed to create NLP as a service with code base for both front end GUI (**`streamlit`**)  and backend server (**`FastApi`**) the usage of transformers models on various downstream NLP task.

The downstream NLP tasks covered:

* News Classification

* Entity Recognition

* Sentiment Analysis

* Summarization

* Information Extraction `To Do`

The user can select different models from the drop down to run the inference.

The users can also directly use the backend fastapi server to have a command line inference. 

<a id='section01a'></a>

### Features of the solution

* **Python Code Base**: Built using `Fastapi` and `Streamlit` making the complete code base in Python.
* **Expandable**: The backend is desinged in a way that it can be expanded with more Transformer based models and it will be available in the front end app automatically. 
* **Micro-Services**: The backend is designed with a microservices architecture, with dockerfile for each service and leveraging on Nginx as a reverse proxy to each independently running service.
    - This makes it easy to update, manitain, start, stop individual NLP services.


<a id='section02'></a>

## Installation

* Clone the Repo.
* Run the `Docker Compose` to spin up the **Fastapi** based backend service.
* Run the **Streamlit app** with the `streamlit run command`.

<a id='section02a'></a>

### Running the Service

1. **Running the backend service.**
    - Go to the `src_fastapi` folder
    - Run the `Docker Compose` comnand

    ```console  
    $ cd src_fastapi
    src_fastapi:~$ sudo docker-compose up -d
    ```

2. **Running the frontend app.**
    <!---
    - Front end is a **`WIP`** as a change in the backend architecture.
    - Should be up in a few days.
    --->
    - Go to the `src_streamlit` folder
    <!---
    - Create the docker image from the `Docker File`
    - Then execute the docker image to spin up a container.
    ```console  
    $ cd src_streamlit
    src_streamlit:~$ sudo docker build -t streamlit_app .
    src_streamlit:~$ sudo docker run -d --name streamlit_app streamlit_app
    ```
    --->
    - Run the app with the streamlit run command
    ```console  
    $ cd src_streamlit
    src_streamlit:~$ streamlit run NLPfily.py
    ```

<a id='section03'></a>

## Project Details

<a id='section03a'></a>

### Demonstration

<p align="center">
<img alt="Project Insight Demo" src="meta/streamlit-NLPfiy.gif">
</p>

<a id='section03b'></a>

### Directory Details

* **Front End**: Front end code is in the `src_streamlit` folder. Along with the `Dockerfile` and `requirements.txt`

* **Back End**: Back End code is in the `src_fastapi` folder.
    * This folder contains directory for each task: `Classification`, `ner`, `summary`...etc
    * Each NLP task has been implemented as a microservice, with its own fastapi server and requirements and Dockerfile so that they can be independently mantained and managed.
    * Each NLP task has its own folder and within each folder each trained model has 1 folder each. For example:
    ```
    - sentiment
        > app
            > api
                > distilbert
                    - model.bin
                    - network.py
                    - tokeniser files
                >roberta
                    - model.bin
                    - network.py
                    - tokeniser files
    ```
    * For each new model under each service a new folder will have to be added.
    * Each folder model will need the following files:
        * Model bin file.
        * Tokenizer files
        * `network.py` Defining the class of the model if customised model used.

    * `config.json`: This file contains the details of the models in the backend and the dataset they are trained on.

<a id='section03c'></a>

### How to Add a new Model

1. Fine Tune a transformer model for specific task. You can leverage the [transformers-tutorials](https://github.com/abhimishra91/transformers-tutorials)

2. Save the model files, tokenizer files and also create a `network.py` script if using a customized training network.

3. Create a directory within the NLP task with `directory_name` as the `model name` and save all the files in this directory.

4. Update the `config.json` with the model details and dataset details.

5. Update the `<service>pro.py` with the correct imports and conditions where the model is imported. For example for a new Bert model in Classification Task, do the following:
    * Create a new directory in `classification/app/api/`. Directory name `bert`.
    * Update `config.json` with following:
        ```
        "classification": {
        "model-1": {
            "name": "DistilBERT",
            "info": "This model is trained on News Aggregator Dataset from UC Irvin Machine Learning Repository. The news headlines are classified into 4 categories: **Business**, **Science and Technology**, **Entertainment**, **Health**. [New Dataset](https://archive.ics.uci.edu/ml/datasets/News+Aggregator)"
        },
        "model-2": {
            "name": "BERT",
            "info": "Model Info"
        }
        }
        ```
    * Update `classificationpro.py` with the following snippets:
        
        **_Only if customized class used_**
        ```
        from classification.bert import BertClass
        ```

        **_Section where the model is selected_**
        ```
        if model == "bert":
            self.model = BertClass()
            self.tokenizer = BertTokenizerFast.from_pretrained(self.path)
        ```

<a id='section04'></a>

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE.md](https://github.com/abhimishra91/insight/blob/master/LICENSE) file for details