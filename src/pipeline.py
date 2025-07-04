from typing import Tuple, Union
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.ingestion import get_data
from src.preprocessing import clean_data, create_features, encode_data, scale_data, split_data
from src.validation import validate_data, version_data
from src.logger import logger

def pipeline(model: str, filename: str = "H1.csv") -> Tuple[DataFrame,DataFrame,Series,Series] | Tuple[DataFrame,DataFrame,Series,Series, Union[StandardScaler, MinMaxScaler]]:
    """
    Runs the full data pipeline: ingest -> clean -> engineer -> validate -> encode/split/scale,
    depending on the model type.

    Args:
        model (str): Model type. Supported values: ['glm', 'gam', 'tree', 'nnet']
        filename (str): The name of the CSV file to load from the extracted ZIP archive. 
                        Defaults to 'H1.csv'.

    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: x_train, x_test, y_train, y_test

    Raises:
        ValueError: If the model type is not supported
        RuntimeError: If data validation fails
    """
    if model not in ['glm', 'gam', 'tree', 'nnet']:
        raise ValueError(f"Unsupported model type: {model}")
    
    try: 
        df_raw = get_data(filename=filename)
        df_cleaned = clean_data(df_raw)
        df_engineered = create_features(df_cleaned)

        if not validate_data(df_engineered):
            raise RuntimeError("Pipeline halted due to failed data validation.")

        if model == "tree":
            df_encoded = encode_data(df_engineered)
            df_encoded.columns = df_encoded.columns.str.strip()
            x_train, x_test, y_train, y_test = split_data(df_encoded, random_state=1234)
            logger.info(f"SUCCESS: Pipeline for {model} succeeded.")
            return x_train, x_test, y_train, y_test
        elif model == "glm" or model == "gam":
            df_encoded = encode_data(df_engineered)
            df_encoded.columns = df_encoded.columns.str.strip()
            x_train, x_test, y_train, y_test = split_data(df_encoded, random_state=1234)
            logger.info(f"SUCCESS: Pipeline for {model} succeeded.")
            return x_train, x_test, y_train, y_test
        elif model == "nnet":
            df_encoded = encode_data(df_engineered)
            df_encoded.columns = df_encoded.columns.str.strip()
            x_train, x_test, y_train, y_test = split_data(df_encoded, random_state=1234)
            x_train, scaler = scale_data(x_train, how="min-max")
            x_test, _ = scale_data(x_test, scaler=scaler)

            # Explicitly cast to float32 to avoid torch.tensor conversion errors
            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")

            logger.info(f"SUCCESS: Pipeline for {model} succeeded.")
            return x_train, x_test, y_train, y_test, scaler

    
    except Exception as e:
        logger.error(f"FAILED: Pipeline failed: {str(e)}")
        raise

