FROM python:3.10-slim

RUN pip install onnxruntime numpy

COPY . /app
WORKDIR /app

CMD ["python", "inference.py"]
