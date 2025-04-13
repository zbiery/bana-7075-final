from src.ingestion import get_data
from src.preprocessing import clean_data, create_features
from src.validation import validate_data, version_data

df_raw = get_data(filename="H1.csv")
df_cleaned = clean_data(df_raw)
df_engineered = create_features(df_cleaned)

valid = validate_data(df_engineered)

if valid:
    version_data()

# print(df_engineered.head(10))
# print(df_engineered.columns)

