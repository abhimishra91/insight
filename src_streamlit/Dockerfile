FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir / src_streamlit

COPY requirements.txt /src_streamlit

WORKDIR /src_streamlit

RUN pip install -r requirements.txt

COPY . /src_streamlit

EXPOSE 8501

CMD ["streamlit", "run", "NLPfiy.py"]

