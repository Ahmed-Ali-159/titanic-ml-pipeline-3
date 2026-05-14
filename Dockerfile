FROM python:3.11-slim

WORKDIR /app

RUN pip install mlflow==2.13.2

ENV MLFLOW_TRACKING_URI=https://dagshub.com/ahmaad.alii.213/titanic-ml-pipeline-3.mlflow

CMD ["mlflow", "models", "serve", \
     "--model-uri", "models:/titanic-classifier@production", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--no-conda"]