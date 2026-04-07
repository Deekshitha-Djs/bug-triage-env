# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Hugging Face model (important for offline run)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Copy all project files
COPY . .

# Expose port (required for Hugging Face)
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
