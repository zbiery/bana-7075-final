from src.ingestion import get_data
from src.preprocessing import clean_data, create_features

df_raw = get_data()
df_cleaned = clean_data(df_raw)
df_final = create_features(df_cleaned, encode=True)

print(df_final.head(10))

