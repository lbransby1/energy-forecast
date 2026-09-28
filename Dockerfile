FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/models ./data/models
COPY data/raw ./data/raw
COPY data/processed/.gitkeep ./data/processed/.gitkeep
COPY data/live/.gitkeep ./data/live/.gitkeep

RUN pip install --no-cache-dir -e .

ENV PORT=8000
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
EXPOSE 8000
CMD ["sh", "-c", "uvicorn energy_forecast.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
