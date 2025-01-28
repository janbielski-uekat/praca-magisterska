# Classification Results Report

## Random Forest

### Best Hyperparameters

**max_depth**: 20 | **min_samples_split**: 2 | **n_estimators**: 100

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8812 |
| Precision for 1  | 0.6721 |
| Recall for 1     | 0.4100 |
| F1-Score for 1   | 0.5093 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 545 | 20 |
| Actual 1 | 59 | 41 |

## Support Vector Machine

### Best Hyperparameters

**C**: 100 | **kernel**: rbf

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8737 |
| Precision for 1  | 0.5833 |
| Recall for 1     | 0.5600 |
| F1-Score for 1   | 0.5714 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 525 | 40 |
| Actual 1 | 44 | 56 |

## Logistic Regression

### Best Hyperparameters

**C**: 10 | **penalty**: l2 | **solver**: lbfgs

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8316 |
| Precision for 1  | 0.4643 |
| Recall for 1     | 0.7800 |
| F1-Score for 1   | 0.5821 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 475 | 90 |
| Actual 1 | 22 | 78 |

## Gradient Boosting

### Best Hyperparameters

**learning_rate**: 0.01 | **max_depth**: 7 | **n_estimators**: 300

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8647 |
| Precision for 1  | 0.5568 |
| Recall for 1     | 0.4900 |
| F1-Score for 1   | 0.5213 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 526 | 39 |
| Actual 1 | 51 | 49 |

## AdaBoost

### Best Hyperparameters

**learning_rate**: 1 | **n_estimators**: 100

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8391 |
| Precision for 1  | 0.4745 |
| Recall for 1     | 0.6500 |
| F1-Score for 1   | 0.5485 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 493 | 72 |
| Actual 1 | 35 | 65 |

## XGBoost

### Best Hyperparameters

**learning_rate**: 0.01 | **max_depth**: 7 | **n_estimators**: 300

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8677 |
| Precision for 1  | 0.5732 |
| Recall for 1     | 0.4700 |
| F1-Score for 1   | 0.5165 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 530 | 35 |
| Actual 1 | 53 | 47 |

## Ridge Classifier

### Best Hyperparameters

**alpha**: 10.0

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8286 |
| Precision for 1  | 0.4583 |
| Recall for 1     | 0.7700 |
| F1-Score for 1   | 0.5746 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 474 | 91 |
| Actual 1 | 23 | 77 |

## SGD Classifier

### Best Hyperparameters

**alpha**: 0.0001 | **l1_ratio**: 0.5 | **penalty**: elasticnet

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8436 |
| Precision for 1  | 0.4857 |
| Recall for 1     | 0.6800 |
| F1-Score for 1   | 0.5667 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 493 | 72 |
| Actual 1 | 32 | 68 |

