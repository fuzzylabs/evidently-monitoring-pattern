FROM python:3.10-slim-buster

WORKDIR /app

RUN pip3 install evidently flask prometheus-client PyYAML pandas

COPY monitoring_server .

CMD ["python", "metric_server.py"]

