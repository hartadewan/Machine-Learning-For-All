FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    mlflow==2.19.0 cloudpickle==3.1.0 pandas==2.2.3 \
    numpy==2.1.3 scikit-learn==1.5.2 scipy==1.14.1 \
    flask==3.1.0 prometheus-client==0.21.1 requests==2.32.3
COPY 7.inference.py 3.prometheus_exporter.py ./

