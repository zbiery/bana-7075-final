# mlflow_server.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Write-Host "Starting MLflow tracking server..."

$projectRoot = (Convert-Path .).Replace('\', '/')
# $projectRoot = "C:/Users/Zbier/source/repos/bana-7075-final"
$artifactRoot = "file:///$projectRoot/mlflow/mlartifacts"
$backendStoreUri = "sqlite:///$projectRoot/mlflow/mlflow.db"

# Create required folders
New-Item -ItemType Directory -Force -Path "$projectRoot/mlflow" | Out-Null
New-Item -ItemType Directory -Force -Path "$projectRoot/mlflow/mlartifacts" | Out-Null

Write-Host "Artifact Root: $artifactRoot"
Write-Host "Backend URI  : $backendStoreUri"
Write-Host "Launching MLflow..."

mlflow server `
  --backend-store-uri $backendStoreUri `
  --default-artifact-root $artifactRoot `
  --host 127.0.0.1 `
  --port 5000

Read-Host "MLflow server started. Press Enter to close..."
