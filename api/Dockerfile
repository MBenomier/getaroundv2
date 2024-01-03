FROM continuumio/miniconda3

WORKDIR  /home/api

RUN apt-get update && apt-get install -y nano unzip && apt-get clean && rm -rf /var/lib/apt/lists/
RUN apt-get update -y 
RUN apt-get install nano unzip
RUN apt-get install -y python3.10
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

COPY . /home/app

CMD gunicorn app:api  --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker 

























