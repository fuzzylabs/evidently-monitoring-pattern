FROM python:3.9-slim-buster

WORKDIR /app

RUN pip3 install flask numpy requests scikit-learn

COPY model_server .

RUN mkdir models

COPY models models

CMD ["python", "inference_server.py"]