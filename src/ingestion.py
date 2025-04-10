import os
import io
import zipfile
import requests
import pandas as pd
from dotenv import load_dotenv
from src.logger import logger 

load_dotenv()
zip_url = os.getenv("data_url")

def get_data(filename: str = "H1.csv") -> pd.DataFrame:
    """
    Downloads a ZIP file from the URL specified in the .env file (under `data_url`),
    extracts its contents, reads the specified CSV file into a pandas DataFrame, 
    and returns the DataFrame.

    Args:
        filename (str): The name of the CSV file to load from the extracted ZIP archive. 
                        Defaults to 'H1.csv'.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of the specified CSV file.

    Raises:
        requests.RequestException: If the file download fails.
        zipfile.BadZipFile: If the downloaded file is not a valid ZIP archive.
        FileNotFoundError: If the specified CSV file is not found after extraction.
        Exception: If reading the CSV into a DataFrame fails.
    """

    logger.info(f"Starting data download from: {zip_url}")

    try:
        response = requests.get(zip_url)
        response.raise_for_status()
        logger.info("Download successful.")
    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise

    extracted_path = "data/raw"
    os.makedirs(extracted_path, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(extracted_path)
            logger.info(f"Extracted all files to: {extracted_path}")
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract ZIP file: {e}")
        raise

    target_file = os.path.join(extracted_path, filename)

    if not os.path.exists(target_file):
        logger.error(f"{filename} not found in extracted files.")
        raise FileNotFoundError(f"{filename} not found after extraction.")

    try:
        df = pd.read_csv(target_file)
        logger.info(f"Successfully loaded {filename} into DataFrame.")
    except Exception as e:
        logger.error(f"Failed to read {filename}: {e}")
        raise

    return df