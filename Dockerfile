# FROM python:3.11-slim

# WORKDIR /app

# RUN pip install mlflow==2.13.2

# ENV MLFLOW_TRACKING_URI=https://dagshub.com/ahmaad.alii.213/titanic-ml-pipeline-3.mlflow

# CMD ["mlflow", "models", "serve", \
#      "--model-uri", "models:/titanic-classifier@production", \
#      "--host", "0.0.0.0", \
#      "--port", "5000", \
#      "--no-conda"]

#######################################################

# Use mlflow-skinny instead of full mlflow - much smaller

FROM python:3.11-slim

WORKDIR /app

# Use mlflow-skinny instead of full mlflow - much smaller
# RUN pip install mlflow-skinny==2.13.2 scikit-learn pandas

# Increase pip timeout to handle slow connections
RUN pip install --default-timeout=300 mlflow-skinny==2.13.2 scikit-learn pandas flask gunicorn

ENV MLFLOW_TRACKING_URI=https://dagshub.com/ahmaad.alii.213/titanic-ml-pipeline-3.mlflow

CMD ["mlflow", "models", "serve", \
     "--model-uri", "models:/titanic-classifier@production", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--no-conda"]