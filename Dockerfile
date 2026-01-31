FROM python:3.13-slim

WORKDIR /Virtual-Assistant.Pipeline

RUN apt-get update && apt-get install -y portaudio19-dev

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /Virtual-Assistant.Pipeline

ENTRYPOINT ["python", "/Virtual-Assistant.Pipeline/llm.py"]

#CMD ["python", "/Virtual-Assistant.Pipeline/llm.py" ]
