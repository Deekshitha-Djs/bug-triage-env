FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Hugging Face model during build so the container runs fully offline
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Copy all project files into the container
COPY . .

# Expose port 7860 for Hugging Face Spaces / API access
EXPOSE 7860

# Default command to run the FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
