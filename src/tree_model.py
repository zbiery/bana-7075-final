import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.ingestion import get_data  # import your team's existing data loader

# 1. Load the dataset (you can switch to "H2.csv" if needed)
df = get_data("H1.csv")

# 2. Define features (X) and target (y)
X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

# 3. Split data with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Initialize and train Decision Tree with 4 key hyperparameters
clf = DecisionTreeClassifier(
    max_depth=5,               # 1. Control overfitting
    min_samples_split=10,      # 2. Minimum samples to split a node
    criterion="entropy",       # 3. Splitting quality function
    max_features="sqrt"        # 4. Number of features considered per split
)
clf.fit(X_train, y_train)

# 5. Predict and evaluate the model
y_pred = clf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
