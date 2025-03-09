FROM python:3.12-slim

WORKDIR /api
# Install build dependencies
RUN apt-get update && apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r /api/requirements.txt

COPY ./api /api

ENV PYTHONPATH=/api

EXPOSE 8505

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8505"]