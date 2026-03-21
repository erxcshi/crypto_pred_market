FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/crypto_pred_market

WORKDIR /app

CMD ["python", "-m", "crypto_pred_market.data_gather.run_all"]
