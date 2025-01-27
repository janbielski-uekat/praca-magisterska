import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from tqdm import tqdm

from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.network import BayesianNetwork
from causalnex.inference import InferenceEngine
from causalnex.evaluation import classification_report
from IPython.lib.display import IFrame
from sklearn.metrics import classification_report as sklearn_classification_report

# Load the data
with open("../../data/processed/modelling_data.pkl", "rb") as file:
    data = joblib.load(file)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]
feature_names = [
    name.replace("Education_2n Cycle", "Education_2n") for name in feature_names
]

# Combine X_train and y_train
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)
train_data = pd.concat([X_train_df, y_train_df], axis=1)
train_data.columns = feature_names + ["Response"]

# Comine X_test and y_test
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
X_test_df = X_test_df.reset_index(drop=True)
y_test_df = pd.Series(y_test).reset_index(drop=True)
test_data = pd.concat([X_test_df, y_test_df], axis=1)
test_data.columns = feature_names + ["Response"]

# Define the target variable and campaign acceptance variables
target_variable = "Response"
campaign_variables = [
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
]

# Define demographic variables
demographic_variables = [
    "Income",
    "CustomerAge",
    "CustomerTenure",
    "HasPartner",
    "Education_2n",
    "Education_Basic",
    "Education_Graduation",
    "Education_Master",
    "Education_PhD",
]

# Automatically generate the edges to exclude
tabu_edges = []

# 1. Response can't influence anything
for feature in feature_names:
    if feature != target_variable:  # Exclude self-loops
        tabu_edges.append((target_variable, feature))

# 2. Campaign acceptance variables can't influence demographics
for campaign_var in campaign_variables:
    for demographic_var in demographic_variables:
        tabu_edges.append((campaign_var, demographic_var))

edges_to_enforce = [
    ("HasPartner", "Response"),
    ("CustomerTenure", "Response"),
    ("MntWines", "Response"),
    ("MntMeatProducts", "Response"),
    ("NumCatalogPurchases", "Response"),
    ("Recency", "Response"),
    ("Teenhome", "Response"),
]
# Create the structure model
sm = from_pandas(train_data, tabu_edges=tabu_edges)

sm.remove_edges_below_threshold(0.5)

for edge in edges_to_enforce:
    sm.add_edge(*edge)

viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
viz.show("../../reports/figures/dags/01_thresholded.html")
IFrame(
    src="../../reports/figures/dags/01_thresholded.html", width="100%", height="100%"
)

sm = sm.get_largest_subgraph()

# Define columns to binarize (threshold at 0.5)
columns_to_binarize = [
    "Education_2n",
    "Education_Basic",
    "Education_Graduation",
    "Education_Master",
    "Education_PhD",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "HasPartner",
]

# Define discretization bins for continuous variables
discretization_bins = {
    "Income": [-np.inf, -0.5, 0, 0.5, np.inf],  # Example bins for normalized Income
    "Recency": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntWines": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntFruits": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntMeatProducts": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntFishProducts": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntSweetProducts": [-np.inf, -0.5, 0, 0.5, np.inf],
    "MntGoldProds": [-np.inf, -0.5, 0, 0.5, np.inf],
    "NumDealsPurchases": [-np.inf, -0.5, 0, 0.5, np.inf],
    "NumWebPurchases": [-np.inf, -0.5, 0, 0.5, np.inf],
    "NumCatalogPurchases": [-np.inf, -0.5, 0, 0.5, np.inf],
    "NumStorePurchases": [-np.inf, -0.5, 0, 0.5, np.inf],
    "NumWebVisitsMonth": [-np.inf, -0.5, 0, 0.5, np.inf],
    "CustomerTenure": [-np.inf, -0.5, 0, 0.5, np.inf],
    "CustomerAge": [-np.inf, -0.5, 0, 0.5, np.inf],
    "Kidhome": [-np.inf, -0.5, 0, 0.5, np.inf],
    "Teenhome": [-np.inf, -0.5, 0, 0.5, np.inf],
}

train_data_discrete = train_data.copy()

# Binarize specified columns in training data
for col in columns_to_binarize:
    if col in train_data_discrete.columns:
        train_data_discrete[col] = (train_data_discrete[col] >= 0.5).astype(int)


# Discretize continuous variables in training data
for feature, bins in discretization_bins.items():
    if feature in train_data_discrete.columns:
        train_data_discrete[feature] = pd.cut(
            train_data_discrete[feature], bins=bins, labels=False
        )

test_data_discrete = test_data.copy()

# Binarize specified columns in test data
for col in columns_to_binarize:
    if col in test_data_discrete.columns:
        test_data_discrete[col] = (test_data_discrete[col] >= 0.5).astype(int)

# Discretize continuous variables in test data
for feature, bins in discretization_bins.items():
    if feature in test_data_discrete.columns:
        test_data_discrete[feature] = pd.cut(
            test_data_discrete[feature], bins=bins, labels=False
        )

# Initialize Bayesian Network from the DAG
bn = BayesianNetwork(sm)  # sm is the DAG from CausalNex

bn = bn.fit_node_states(train_data_discrete)

bn = bn.fit_cpds(train_data_discrete, method="BayesianEstimator", bayes_prior="K2")

# Validate the learned CPDs
cpds_summary = {node: bn.cpds[node] for node in bn.nodes}

# Evaluate the network using the test dataset
ie = InferenceEngine(bn)

classification_report(bn, test_data_discrete, "Response")

predicted_probabilities = bn.predict_probability(test_data_discrete, "Response")

y_pred = predicted_probabilities["Response_1"].apply(lambda x: 1 if x >= 0.5 else 0)

print(sklearn_classification_report(test_data_discrete["Response"], y_pred))
