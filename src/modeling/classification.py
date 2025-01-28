import json
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from reporting import generate_markdown_report

# Load the data
with open("../../data/processed/modelling_data.pkl", "rb") as file:
    data = joblib.load(file)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Initialize classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    "XGBoost": XGBClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(),
}

# Define hyperparameters for grid search
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "Ridge Classifier": {
        "alpha": [0.1, 1.0, 10.0],
    },
    "SGD Classifier": {
        "alpha": [0.0001, 0.001, 0.01],
        "l1_ratio": [0.15, 0.5, 0.85],
        "penalty": ["l2", "l1", "elasticnet"],
    },
}

best_params = {}

# Perform grid search for each classifier
best_classifiers = {}
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring="recall")
    grid_search.fit(X_train, y_train)
    best_classifiers[name] = grid_search.best_estimator_
    best_params[name] = grid_search.best_params_
    print(f"Best parameters for {name}: {grid_search.best_params_}")

# Update classifiers with the best estimators
classifiers = best_classifiers

results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results.append(
        {
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": report["1"]["precision"],  # Precision for positive class
            "Recall": report["1"]["recall"],  # Recall for positive class
            "F1-Score": report["1"]["f1-score"],  # F1-Score for positive class
            "Confusion Matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
            "Best Hyperparameters": best_params[
                name
            ],  # Add best hyperparameters to results
        }
    )

    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
    print(f"Confusion Matrix:\n{conf_matrix}\n")

# Save all trained classifiers to a single file
joblib.dump(classifiers, "../../models/trained_classifiers.pkl")

# Save results to a JSON file
with open("../../reports/results/classification_results.json", "w") as f:
    json.dump(results, f, indent=4)

generate_markdown_report(
    "../../reports/results/classification_results.json",
    "../../reports/clf_reports/classification_results_report.md",
)
