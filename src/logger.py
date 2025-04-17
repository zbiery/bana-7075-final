import logging
import os
import pandas as pd
import json
import datetime

LOG_DIR = "logs"
REQUESTS_LOG_PATH = "logs/prediction_logs.csv"

# Set up basic system logging
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "running_logs.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("src")

# Set up prediction/request logging
if not os.path.exists(REQUESTS_LOG_PATH):
    pd.DataFrame(columns=[
        "timestamp", "request_type", "input_data", "predictions", "status"
    ]).to_csv(REQUESTS_LOG_PATH, index=False)

# Primary function to log inputs & predictions
def log_request(request_type, input_data, predictions, probs, status):
    """Logs the request details to a CSV file."""
    log_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "request_type": request_type,
        "input_data": json.dumps(input_data),
        "predictions": json.dumps(predictions),
        "probabilities": json.dumps(probs),
        "status": status
    }])
    log_entry.to_csv(REQUESTS_LOG_PATH, mode='a', header=False, index=False)