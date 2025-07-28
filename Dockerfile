FROM python:3.11-slim

# Install build dependencies for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libmupdf-dev gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY approach_explanation.md .
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]