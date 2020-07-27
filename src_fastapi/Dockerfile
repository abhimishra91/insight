  
FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /src_fastapi

COPY requirements.txt /src_fastapi

WORKDIR /src_fastapi

RUN pip install -r requirements.txt

COPY . /src_fastapi

EXPOSE 8000

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]