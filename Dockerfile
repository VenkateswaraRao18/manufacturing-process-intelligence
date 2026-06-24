FROM python:3.11-slim

WORKDIR /app

# System deps for umap-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn streamlit

COPY . .

# Train models if not present
RUN python main.py || true

EXPOSE 8000 8501

# Default: start API. Override CMD for dashboard.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
