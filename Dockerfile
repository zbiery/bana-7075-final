# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 1000
EXPOSE 8501
EXPOSE 5000

# Run FastAPI server by default
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "1000"]
