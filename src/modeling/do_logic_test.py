import joblib
import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork as PGM_BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from causalnex.structure.notears import from_pandas
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.metrics import RocCurveDisplay

# --- STEP 1: Load modeling data ---
# Load the data
with open("../../data/processed/bn_modelling_data.pkl", "rb") as file:
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
X_train_df = pd.DataFrame(X_train).reset_index(drop=True)
y_train_df = pd.DataFrame(y_train).reset_index(drop=True)
train_data = pd.concat([X_train_df, y_train_df], axis=1)
train_data.columns = feature_names + ["Response"]

# Comine X_test and y_test
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
X_test_df = X_test_df.reset_index(drop=True)
y_test_df = pd.Series(y_test).reset_index(drop=True)
test_data = pd.concat([X_test_df, y_test_df], axis=1)
test_data.columns = feature_names + ["Response"]

# --- STEP 2: Define discretization ---
binary_columns = [
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


def discretize_data(X, bin_edges=None):
    X_discrete = X.copy()
    new_bin_edges = {}

    for col in binary_columns:
        X_discrete[col] = (X_discrete[col] >= 0.5).astype(int)

    for col in X.columns:
        if col in binary_columns or col == "Response":
            continue
        if bin_edges and col in bin_edges:
            bins = bin_edges[col]
        else:
            _, bins = pd.qcut(X[col], q=3, retbins=True, duplicates="drop")
        X_discrete[col] = pd.cut(X[col], bins=bins, labels=False, include_lowest=True)
        new_bin_edges[col] = bins

    return X_discrete, new_bin_edges


# --- STEP 3: Discretize train and test data ---
train_data_discrete, bin_edges = discretize_data(train_data)

test_data_discrete, _ = discretize_data(test_data, bin_edges)

# --- STEP 4: Learn DAG and fit Bayesian Network ---
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

# 2. Campaign acceptance variables can't influence anything except target variable
for feature in campaign_variables:
    if feature != target_variable:  # Exclude self-loops
        for other_feature in feature_names:
            if other_feature != target_variable and other_feature != feature:
                tabu_edges.append((feature, other_feature))


edges_to_enforce = [
    ("HasPartner", "Response"),
    ("CustomerTenure", "Response"),
    ("MntMeatProducts", "Response"),
    ("Recency", "Response"),
    ("Teenhome", "Response"),
]

sm = from_pandas(train_data_discrete, tabu_edges=tabu_edges)
sm.remove_edges_below_threshold(0.5)
for edge in edges_to_enforce:
    sm.add_edge(*edge)
sm = sm.get_largest_subgraph()
edges = list(sm.edges)

# Visualize the structure model with directed edges
nx_graph = nx.DiGraph(sm.edges)

# Define node positions using a circular layout
pos = nx.circular_layout(nx_graph)  # Arrange nodes in a circle

# Customize node and edge styles
node_colors = "lightblue"
node_size = 3000  # Increased for better visibility
font_size = 10
edge_color = "gray"
arrowstyle = "-|>"  # Ensure arrows indicate direction
arrowsize = 20  # Increase arrow size for better visibility

# Draw the graph
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=node_size)
nx.draw_networkx_labels(nx_graph, pos, font_size=font_size, font_color="black")
nx.draw_networkx_edges(
    nx_graph,
    pos,
    edge_color=edge_color,
    arrowstyle=arrowstyle,
    arrowsize=arrowsize,
    connectionstyle="arc3,rad=0.5",  # Add slight curvature for better clarity
    arrows=True,  # Enable arrows for directed edges
    node_size=node_size,  # Account for node size to position arrows correctly
)

# Add a title
plt.title("Bayesian Network Structure", fontsize=16)

# Remove axes for a cleaner look
plt.axis("off")
plt.tight_layout()
plt.show()


model = PGM_BayesianNetwork(edges)
model.fit(train_data_discrete, estimator=BayesianEstimator, prior_type="BDeu")
infer = VariableElimination(model)


# --- STEP 5: Evaluate on test set ---
def evaluate_pgmpy_model(model, infer, X_test, y_test, target="Response"):
    preds = []
    model_nodes = set(model.nodes())

    for _, row in X_test.iterrows():
        evidence = {k: v for k, v in row.to_dict().items() if k in model_nodes}
        try:
            dist = infer.query(variables=[target], evidence=evidence).values
            prob_1 = dist[1] if len(dist) > 1 else 0.0
            preds.append(prob_1)
        except Exception:
            preds.append(np.nan)

    y_test = y_test.loc[X_test.index]
    valid_idx = ~np.isnan(preds)
    y_true = y_test[valid_idx]
    y_scores = np.array(preds)[valid_idx]
    y_pred_class = (y_scores > 0.25).astype(int)

    cm = confusion_matrix(y_true, y_pred_class)
    cls_report = classification_report(y_true, y_pred_class, output_dict=True)

    metrics = {
        # "ROC AUC": roc_auc_score(y_true, y_scores),
        "Accuracy (0.25 threshold)": accuracy_score(y_true, y_pred_class),
        "Brier Score": brier_score_loss(y_true, y_scores),
        "Valid predictions": len(y_scores),
        "Total test samples": len(y_test),
        "Confusion Matrix": cm.tolist(),
        "Classification Report": cls_report,  # as list so it's JSON/print safe
    }

    # Create a figure with two subplots side by side
    sns.set_context("notebook", font_scale=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot confusion matrix on the left
    cm = confusion_matrix(y_true, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicted", fontsize=16)
    axes[0].set_ylabel("True", fontsize=16)
    axes[0].set_title("Confusion Matrix for Bayesian Network", fontsize=18)

    # Plot ROC curve on the right
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=axes[1])
    axes[1].set_title("ROC Curve for Bayesian Network", fontsize=18)

    # Save the combined figure
    plt.tight_layout()
    plt.savefig("../../reports/figures/confusion_matrix_roc_curve_bn.png")
    plt.close()

    return metrics


# df_effect = estimate_causal_effect_pgmpy(model, train_data_discrete, feature="Recency")


# ==========================================================================================
def _p1_or_expectation(dist_values):
    # Jeśli target binarny: zwróć P(Y=1). Jeśli wielowartościowy: zwróć E[Y].
    values = np.asarray(dist_values).ravel()
    if values.size == 2:
        return float(values[1])
    # E[Y] dla indeksów 0..K-1
    idx = np.arange(values.size, dtype=float)
    return float(np.dot(idx, values))


def estimate_do_gformula(model, data_discrete, feature, target="Response", plot=True):
    """
    Szacuje P(target | do(feature=val)) przez g-formułę (uśrednianie po p(Z), gdzie Z=Pa(feature)).
    Nie warunkujemy na potomkach X (unik post-treatment bias).
    """
    infer = VariableElimination(model)
    values = sorted(data_discrete[feature].dropna().unique())

    # Zbiór regulujący: rodzice X (są wystarczający do back-door w DAG bez ukrytych)
    adjust_set = model.get_parents(feature)

    results = {}
    if len(adjust_set) == 0:
        # Brak regulacji: P(Y|do(X=x)) = sum_y ...; w praktyce: P(Y|X=x) uśrednione po populacji Z=puste
        for val in values:
            q = infer.query(
                variables=[target], evidence={feature: int(val)}, show_progress=False
            )
            results[val] = _p1_or_expectation(q.values)
    else:
        # Standardization: średnia po empirycznym p(z)
        Z_df = data_discrete[adjust_set]
        for val in values:
            probs = []
            for _, zrow in Z_df.iterrows():
                evidence = {feature: int(val)}
                evidence.update({z: int(zrow[z]) for z in adjust_set})
                try:
                    q = infer.query(
                        variables=[target], evidence=evidence, show_progress=False
                    )
                    probs.append(_p1_or_expectation(q.values))
                except Exception:
                    continue
            results[val] = float(np.mean(probs)) if probs else np.nan

    df = pd.DataFrame.from_dict(
        results, orient="index", columns=[f"P({target}|do({feature}))"]
    )
    df.index.name = feature
    df = df.sort_index()

    if plot:
        ax = df.plot(kind="bar", legend=False)
        ax.set_title(f"Total Effect (g-formula): do({feature}=·) → {target}")
        ax.set_xlabel(f"{feature} (discretized levels)")
        ax.set_ylabel(
            f"P({target}=1)"
            if len(model.get_cpds(target).values) == 2
            else f"E[{target}]"
        )
        plt.axhline(
            data_discrete[target].mean()
            if data_discrete[target].nunique() == 2
            else None,
            linestyle="--",
            color="red",
        )
        plt.tight_layout()
        plt.grid(True)
        plt.show()
    return df


def rank_causal_features(
    model,
    data_discrete,
    features_to_rank,
    target="Response",
):
    """
    Ranguje cechy na podstawie ich całkowitego wpływu przyczynowego (Total Effect)
    na zmienną docelową, wykorzystując marginalne interwencje.

    Args:
        model: Nauczony model sieci bayesowskiej.
        data_discrete: Dyskretyzowane dane treningowe.
        features_to_rank: Lista nazw cech do oceny.
        target: Nazwa zmiennej docelowej.

    Returns:
        DataFrame z rankingiem cech i miarami wpływu.
    """
    infer = VariableElimination(model)
    target_mean = data_discrete[target].mean()
    results = {}

    for feature in features_to_rank:
        values = sorted(data_discrete[feature].dropna().unique())
        adjust_set = model.get_parents(feature)
        effect_projections = []

        if len(adjust_set) == 0:
            for val in values:
                q = infer.query(
                    variables=[target],
                    evidence={feature: int(val)},
                    show_progress=False,
                )
                effect_projections.append(_p1_or_expectation(q.values))
        else:
            Z_df = data_discrete[adjust_set]
            for val in values:
                probs = []
                for _, zrow in Z_df.iterrows():
                    evidence = {feature: int(val)}
                    evidence.update({z: int(zrow[z]) for z in adjust_set})
                    try:
                        q = infer.query(
                            variables=[target], evidence=evidence, show_progress=False
                        )
                        probs.append(_p1_or_expectation(q.values))
                    except Exception:
                        continue
                if probs:
                    effect_projections.append(np.mean(probs))
                else:
                    effect_projections.append(np.nan)

        # Obliczamy "Avg. Shift in Expected Response"
        effect_projections = np.array(effect_projections)
        # Zastąp NaN średnią wartością z pozostałych projekcji
        effect_projections[np.isnan(effect_projections)] = np.nanmean(
            effect_projections
        )
        avg_shift = np.mean(np.abs(effect_projections - target_mean))
        results[feature] = avg_shift

    df_rank = pd.DataFrame.from_dict(
        results, orient="index", columns=["Avg. Shift in Expected Response"]
    )
    df_rank = df_rank.sort_values(by="Avg. Shift in Expected Response", ascending=True)

    return df_rank


sns.set_context("notebook", font_scale=1)

# G-formuła (regulacja przez rodziców X)
estimate_do_gformula(model, train_data_discrete, feature="Recency")
estimate_do_gformula(model, train_data_discrete, feature="Teenhome")
estimate_do_gformula(model, train_data_discrete, feature="MntWines")
estimate_do_gformula(model, train_data_discrete, feature="CustomerTenure")

metrics = evaluate_pgmpy_model(
    model,
    infer,
    test_data_discrete.drop(columns="Response"),
    test_data_discrete["Response"],
)

print(metrics)

# --- STEP 6: Rank features based on causal influence ---

# Wybierz cechy do rankingu.
features_to_rank = [node for node in model.nodes() if node != "Response"]

# Generuj ranking
causal_ranking = rank_causal_features(
    model,
    train_data_discrete,
    features_to_rank=features_to_rank,
    target="Response",
)

print("\n--- Feature Ranking based on Causal Influence ---")
print(causal_ranking)

# Wizualizuj ranking
plt.figure(figsize=(10, 8))
# Sort in descending order for visualization (reverse order)
causal_ranking_reversed = causal_ranking.sort_values(
    by="Avg. Shift in Expected Response", ascending=False
)
sns.barplot(
    x="Avg. Shift in Expected Response",
    y=causal_ranking_reversed.index,
    data=causal_ranking_reversed,
    color="steelblue",
)
plt.title("Total Effect on Response (Marginal Interventions)", fontsize=16)
plt.xlabel("Avg. Shift in Expected Response")
plt.ylabel("")  # Pusta etykieta osi y
plt.tight_layout()
plt.show()
