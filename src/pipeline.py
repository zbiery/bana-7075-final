from typing import Tuple
from pandas import DataFrame, Series
from src.ingestion import get_data
from src.preprocessing import clean_data, create_features, encode_data, scale_data, split_data
from src.validation import validate_data, version_data
from src.logger import logger

def pipeline(model: str) -> Tuple[DataFrame,DataFrame,Series,Series]:
    """
    Runs the full data pipeline: ingest -> clean -> engineer -> validate -> encode/split/scale,
    depending on the model type.

    Args:
        model (str): Model type. Supported values: ['glm', 'gam', 'tree', 'nnet']

    Returns:
        Tuple[DataFrame, DataFrame, Series, Series]: x_train, x_test, y_train, y_test

    Raises:
        ValueError: If the model type is not supported
        RuntimeError: If data validation fails
    """
    if model not in ['glm', 'gam', 'tree', 'nnet']:
        raise ValueError(f"Unsupported model type: {model}")
    
    try: 
        df_raw = get_data(filename="H1.csv")
        df_cleaned = clean_data(df_raw)
        df_engineered = create_features(df_cleaned)

        if not validate_data(df_engineered):
            raise RuntimeError("Pipeline halted due to failed data validation.")

        if model == "tree":
            x_train, x_test, y_train, y_test = split_data(df_engineered, random_state=1234)
        elif model == "glm" or model == "gam":
            df_encoded = encode_data(df_engineered)
            x_train, x_test, y_train, y_test = split_data(df_encoded, random_state=1234)
        else:
            df_encoded = encode_data(df_engineered)
            x_train, x_test, y_train, y_test = split_data(df_encoded, random_state=1234)
            x_train = scale_data(x_train, how = "min-max")
            x_test = scale_data(x_test, how="min-max")

        logger.info(f"SUCCESS: Pipeline succeeded.")
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"FAILED: Pipeline failed: {str(e)}")
        raise

