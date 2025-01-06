import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
raw_file_path = "../data/raw/original_data.csv"
raw_data = pd.read_csv(raw_file_path, parse_dates=["Dt_Customer"])

# Drop rows with missing data
cleaned_data = raw_data.dropna().copy()

# Calculate "CustomerTenure" from Dt_Customer using the maximum date in the column + one day
max_date = cleaned_data["Dt_Customer"].max()
cleaned_data["CustomerTenure"] = (
    max_date + pd.Timedelta(days=1) - cleaned_data["Dt_Customer"]
).dt.days

# Calculate CustomerAge from Year_Birth assuming max_date is the current date
cleaned_data["CustomerAge"] = max_date.year - cleaned_data["Year_Birth"]

# Simplify Marital_Status to HasPartner
partner_status = {
    "Married": 1,
    "Together": 1,
    "Single": 0,
    "Divorced": 0,
    "Widow": 0,
    "Alone": 0,
    "Absurd": 0,
    "YOLO": 0,
}
cleaned_data["HasPartner"] = cleaned_data["Marital_Status"].map(partner_status)

# Drop redundant columns
cleaned_data.drop(
    columns=[
        "ID",
        "Dt_Customer",
        "Year_Birth",
        "Marital_Status",
        "Z_CostContact",
        "Z_Revenue",
    ],
    inplace=True,
)

# Save the cleaned dataset for exploratory data analysis
cleaned_data.to_pickle("../data/processed/exploration_data.pkl")

# Separate features and target variable
X = cleaned_data.drop(columns=["Response"])
y = cleaned_data["Response"]

# Infer data types
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
binary_cols = [col for col in numerical_cols if X[col].nunique() == 2]

# Remove binary columns from numerical and categorical lists
numerical_cols = [col for col in numerical_cols if col not in binary_cols]
categorical_cols = [col for col in categorical_cols if col not in binary_cols]

# Define preprocessing pipelines for numerical, categorical and binary data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

binary_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("binary", binary_transformer, binary_cols),
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply preprocessing to training and testing data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE to the preprocessed training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Create a dictionary to hold the modelling data
modelling_data = {
    "X_train": X_train_resampled,
    "y_train": y_train_resampled,
    "X_test": X_test_preprocessed,
    "y_test": y_test,
}

# Save the modelling data
joblib.dump(modelling_data, "../data/processed/modelling_data.pkl")
