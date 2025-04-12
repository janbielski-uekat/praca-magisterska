import json
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

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
        "learning_rate": [0.01, 0.1, 1],
        "max_depth": [None, 10, 20, 30],
    },
    "AdaBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 1],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 1],
        "max_depth": [None, 10, 20, 30],
    },
    "Ridge Classifier": {
        "alpha": [0.1, 1.0, 10.0],
    },
    "SGD Classifier": {
        "alpha": [0.0001, 0.001, 0.01],
        "l1_ratio": [0.15, 0.5, 0.85],
        "penalty": ["l2", "l1", "elasticnet"],
        "loss": ["hinge", "log", "squared_hinge"],
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

conversion_return = 10.93  # Example: profit per successful conversion
contact_cost = 3

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate profit
    true_positives = conf_matrix[1][
        1
    ]  # True Positives (bottom-right of confusion matrix)
    total_predicted_positives = sum(
        conf_matrix[:, 1]
    )  # Total Predicted Positives (second column)
    profit = (true_positives * conversion_return) - (
        total_predicted_positives * contact_cost
    )

    results.append(
        {
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": report["1"]["precision"],  # Precision for positive class
            "Recall": report["1"]["recall"],  # Recall for positive class
            "F1-Score": report["1"]["f1-score"],  # F1-Score for positive class
            "Profit": profit,
            "Confusion Matrix": conf_matrix.tolist(),  # Convert numpy array to list for JSON serialization
            "Best Hyperparameters": best_params[
                name
            ],  # Add best hyperparameters to results
        }
    )

    print(f"Classifier: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")

    # Create a figure with two subplots side by side
    sns.set_context("notebook", font_scale=1.5)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot confusion matrix on the left
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicted", fontsize=16)
    axes[0].set_ylabel("True", fontsize=16)
    axes[0].set_title(f"Confusion Matrix for {name}", fontsize=18)

    # Plot ROC curve on the right
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"ROC Curve for {name}", fontsize=18)

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/confusion_matrix_roc_curve_{name}.png")
    plt.close()


# Save all trained classifiers to a single file
joblib.dump(classifiers, "../../models/trained_classifiers.pkl")


# Save results to a JSON file
with open("../../reports/results/classification_results.json", "w") as f:
    json.dump(results, f, indent=4)

generate_markdown_report(
    "../../reports/results/classification_results.json",
    "../../reports/clf_reports/classification_results_report.md",
)
