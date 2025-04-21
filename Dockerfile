# Food & Weather Buddy – Dockerfile
FROM python:3.11-slim

# Avoid interactive prompts & byte‑code files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Core dependencies (no GPU required)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Launch API
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
