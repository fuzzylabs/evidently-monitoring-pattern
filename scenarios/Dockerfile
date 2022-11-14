FROM python:3.9-slim-buster

WORKDIR /app

RUN pip3 install pandas requests

COPY scenarios .

RUN mkdir datasets

COPY datasets datasets

ENTRYPOINT ["python", "scenario_simulator.py" ]
