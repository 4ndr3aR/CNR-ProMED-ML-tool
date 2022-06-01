FROM python:3.8.13-slim-bullseye

RUN apt-get update && apt-get install -y git python3-dev gcc procps htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -U pip wheel setuptools
RUN pip install --upgrade -r requirements.txt

COPY app app/

ENV EXT_USERNAME=extuser1
ENV EXT_PASSWORD=extpassword1

RUN python app/server.py

EXPOSE 55573 55574

CMD ["python", "app/server.py", "--cmd", "serve", "--model-url", "http://http://deeplearning.ge.imati.cnr.it/promed/models/promed-multioutput-regression-problem-v0.1-endoftraining.pkl", "--model-name", "promed-multioutput-regression-problem-v0.1-endoftraining.pkl", "--flask-port", "55573", "--web-port", "55574"]
