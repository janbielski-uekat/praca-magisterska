# Classification Results Report

## Random Forest

### Best Hyperparameters

**max_depth**: 30 | **min_samples_split**: 2 | **n_estimators**: 100

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8842 |
| Precision for 1  | 0.6825 |
| Recall for 1     | 0.4300 |
| F1-Score for 1   | 0.5276 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 545 | 20 |
| Actual 1 | 57 | 43 |

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
| Accuracy   | 0.8632 |
| Precision for 1  | 0.5517 |
| Recall for 1     | 0.4800 |
| F1-Score for 1   | 0.5134 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 526 | 39 |
| Actual 1 | 52 | 48 |

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

**alpha**: 0.001 | **l1_ratio**: 0.5 | **loss**: hinge | **penalty**: l2

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8150 |
| Precision for 1  | 0.4378 |
| Recall for 1     | 0.8100 |
| F1-Score for 1   | 0.5684 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 461 | 104 |
| Actual 1 | 19 | 81 |

