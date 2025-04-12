# Classification Results Report

## Random Forest

### Best Hyperparameters

**max_depth**: 30 | **min_samples_split**: 2 | **n_estimators**: 300

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8887 |
| Precision for 1  | 0.6970 |
| Recall for 1     | 0.4600 |
| F1-Score for 1   | 0.5542 |
| Profit           | 304.7800 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 545 | 20 |
| Actual 1 | 54 | 46 |

## Support Vector Machine

### Best Hyperparameters

**C**: 100 | **kernel**: rbf

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8662 |
| Precision for 1  | 0.5579 |
| Recall for 1     | 0.5300 |
| F1-Score for 1   | 0.5436 |
| Profit           | 294.2900 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 523 | 42 |
| Actual 1 | 47 | 53 |

## Logistic Regression

### Best Hyperparameters

**C**: 1 | **penalty**: l2 | **solver**: lbfgs

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8331 |
| Precision for 1  | 0.4675 |
| Recall for 1     | 0.7900 |
| F1-Score for 1   | 0.5874 |
| Profit           | 356.4700 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 475 | 90 |
| Actual 1 | 21 | 79 |

## Gradient Boosting

### Best Hyperparameters

**learning_rate**: 1 | **max_depth**: 10 | **n_estimators**: 200

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8737 |
| Precision for 1  | 0.5930 |
| Recall for 1     | 0.5100 |
| F1-Score for 1   | 0.5484 |
| Profit           | 299.4300 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 530 | 35 |
| Actual 1 | 49 | 51 |

## AdaBoost

### Best Hyperparameters

**learning_rate**: 1 | **n_estimators**: 100

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8556 |
| Precision for 1  | 0.5159 |
| Recall for 1     | 0.6500 |
| F1-Score for 1   | 0.5752 |
| Profit           | 332.4500 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 504 | 61 |
| Actual 1 | 35 | 65 |

## XGBoost

### Best Hyperparameters

**learning_rate**: 0.1 | **max_depth**: 20 | **n_estimators**: 100

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8872 |
| Precision for 1  | 0.6582 |
| Recall for 1     | 0.5200 |
| F1-Score for 1   | 0.5810 |
| Profit           | 331.3600 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 538 | 27 |
| Actual 1 | 48 | 52 |

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
| Profit           | 337.6100 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 474 | 91 |
| Actual 1 | 23 | 77 |

## SGD Classifier

### Best Hyperparameters

**alpha**: 0.001 | **l1_ratio**: 0.85 | **loss**: hinge | **penalty**: elasticnet

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8075 |
| Precision for 1  | 0.4293 |
| Recall for 1     | 0.8500 |
| F1-Score for 1   | 0.5705 |
| Profit           | 335.0500 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 452 | 113 |
| Actual 1 | 15 | 85 |

