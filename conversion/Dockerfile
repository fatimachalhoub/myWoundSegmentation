
FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip &&\
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "conversion.py"]
