# 1. Base image
FROM python:3.9-slim

# 2. Copy files
COPY . /src

# 3. Install dependencies
RUN pip install -r /src/requirements.txt
RUN pip install streamlit
RUN which streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "prompt-select-streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]