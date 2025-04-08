import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

def run_decision_tree(file_path, output_image):
    print(f"\n--- Running model for {file_path} ---")

    # 1. Load data
    df = pd.read_csv(file_path)

    # 2. Show column names
    print("Columns in the dataset:", df.columns)

    # 3. Convert categorical variables to dummy/indicator variables
    df = pd.get_dummies(df)

    # 4. Define features (X) and target (y)
    X = df.drop("IsCanceled", axis=1)
    y = df["IsCanceled"]

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Train the Decision Tree
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        criterion="entropy",
        max_features="sqrt"
    )
    clf.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 8. Save decision tree plot
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=["Not Canceled", "Canceled"],
              filled=True, max_depth=3, fontsize=8)
    plt.title(f"Decision Tree for {file_path}")
    plt.savefig(output_image)
    plt.close()


# Run the model for both datasets and save results
run_decision_tree("data/H1.csv", "output/tree_H1.png")
run_decision_tree("data/H2.csv", "output/tree_H2.png")
