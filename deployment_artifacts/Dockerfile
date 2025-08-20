
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY darkoperator/ ./darkoperator/
COPY pyproject.toml .

RUN pip install -e .

EXPOSE 8080

CMD ["python", "-m", "darkoperator.cli.main", "--port", "8080"]
