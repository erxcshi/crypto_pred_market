FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements-collector.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-collector.txt

COPY . .

CMD ["python", "-m", "data_gather.scrapers.run_all"]
