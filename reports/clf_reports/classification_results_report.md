# Classification Results Report

## Random Forest

### Best Hyperparameters

**max_depth**: 20 | **min_samples_split**: 2 | **n_estimators**: 300

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8887 |
| Precision for 1  | 0.6857 |
| Recall for 1     | 0.4800 |
| F1-Score for 1   | 0.5647 |
| ROAS           | 2.4983 |
| Profit        | 314.6400 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 543 | 22 |
| Actual 1 | 52 | 48 |

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
| ROAS           | 2.0326 |
| Profit        | 294.2900 |

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
| ROAS           | 1.7031 |
| Profit        | 356.4700 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 475 | 90 |
| Actual 1 | 21 | 79 |

## Gradient Boosting

### Best Hyperparameters

**learning_rate**: 0.1 | **max_depth**: 10 | **n_estimators**: 300

### Classification Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8707 |
| Precision for 1  | 0.5946 |
| Recall for 1     | 0.4400 |
| F1-Score for 1   | 0.5057 |
| ROAS           | 2.1663 |
| Profit        | 258.9200 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 535 | 30 |
| Actual 1 | 56 | 44 |

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
| ROAS           | 1.8795 |
| Profit        | 332.4500 |

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
| ROAS           | 2.3981 |
| Profit        | 331.3600 |

### Confusion Matrix

|   | Predicted 0 | Predicted 1 |
|---|--------------|--------------|
| Actual 0 | 538 | 27 |
| Actual 1 | 48 | 52 |

