  
# Getting the Base Image
FROM tiangolo/uvicorn-gunicorn:python3.7

# Creating a new folder
RUN mkdir /src_fastapi

# Copy the requirements file
COPY requirements.txt /src_fastapi

# Change the working directory
WORKDIR /src_fastapi

# Installing Packages
RUN pip install -r requirements.txt

# Copy everything to working directory
COPY . /src_fastapi

# Exposing the port
EXPOSE 8000

# The server will run on: http://172.17.0.2:8000
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]