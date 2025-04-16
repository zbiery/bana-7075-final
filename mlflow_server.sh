#!/bin/bash

set -e  # Exit on error

echo "Starting MLflow tracking server..."

# Get the current project root directory
project_root=$(pwd)
artifact_root="file://$project_root/mlflow/mlartifacts"
backend_store_uri="sqlite:///$project_root/mlflow/mlflow.db"

# Create required folders
mkdir -p "$project_root/mlflow/mlartifacts"

echo "Artifact Root: $artifact_root"
echo "Backend URI  : $backend_store_uri"
echo "Launching MLflow..."

# Run the MLflow server
mlflow server \
  --backend-store-uri "$backend_store_uri" \
  --default-artifact-root "$artifact_root" \
  --host 127.0.0.1 \
  --port 5000
