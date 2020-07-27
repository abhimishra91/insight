# Getting the Base Image
FROM tiangolo/uvicorn-gunicorn:python3.7

# Creating a new folder
RUN mkdir /src_streamlit

# Copy the requirements file
COPY requirements.txt /src_streamlit

# Change the working directory
WORKDIR /src_streamlit

# Installing Packages
RUN pip install -r requirements.txt

# Copy everything to working directory
COPY . /src_streamlit

# Exposing the port
EXPOSE 8501

# Running the streamlit service
CMD ["streamlit", "run", "NLPfiy.py"]