import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def generate_markdown_report(json_file_path, markdown_file_path):
    """
    Generate a Markdown report from the classification results JSON file.

    Parameters:
    - json_file_path: str, path to the JSON file containing classification results.
    - markdown_file_path: str, path to the output Markdown file.
    """
    # Load the classification results from the JSON file
    with open(json_file_path, "r") as f:
        results = json.load(f)

    # Generate Markdown content
    markdown_content = "# Classification Results Report\n\n"

    for result in results:
        markdown_content += f"## {result['Classifier']}\n\n"
        markdown_content += "### Best Hyperparameters\n\n"
        hyperparameters = " | ".join(
            [
                f"**{param}**: {value}"
                for param, value in result["Best Hyperparameters"].items()
            ]
        )
        markdown_content += f"{hyperparameters}\n\n"

        # Add classification metrics in a table format
        markdown_content += "### Classification Metrics\n\n"
        markdown_content += "| Metric     | Value   |\n"
        markdown_content += "|------------|---------|\n"
        markdown_content += f"| Accuracy   | {result['Accuracy']:.4f} |\n"
        markdown_content += f"| Precision for 1  | {result['Precision']:.4f} |\n"
        markdown_content += f"| Recall for 1     | {result['Recall']:.4f} |\n"
        markdown_content += f"| F1-Score for 1   | {result['F1-Score']:.4f} |\n"
        markdown_content += f"| ROAS           | {result['ROAS']:.4f} |\n"
        markdown_content += f"| Profit        | {result['Profit']:.4f} |\n"
        markdown_content += "\n"

        markdown_content += "### Confusion Matrix\n\n"
        markdown_content += "|   | Predicted 0 | Predicted 1 |\n"
        markdown_content += "|---|--------------|--------------|\n"
        markdown_content += f"| Actual 0 | {result['Confusion Matrix'][0][0]} | {result['Confusion Matrix'][0][1]} |\n"
        markdown_content += f"| Actual 1 | {result['Confusion Matrix'][1][0]} | {result['Confusion Matrix'][1][1]} |\n"
        markdown_content += "\n"

    # Write the Markdown content to a file
    with open(markdown_file_path, "w") as f:
        f.write(markdown_content)

    print("Markdown report generated successfully.")


def generate_classifier_ranking_plots(json_file_path, output_dir):
    """
    Generate bar plots showing classifier rankings for different metrics.

    Parameters:
    - json_file_path: str, path to the JSON file containing classification results.
    - output_dir: str, directory to save the ranking plots.
    """
    # Load the classification results from the JSON file
    with open(json_file_path, "r") as f:
        results = json.load(f)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Define metrics to plot (excluding confusion matrix and hyperparameters)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROAS", "Profit"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plot style
    plt.style.use("default")
    single_color = "#2E86AB"  # Single blue color for all bars

    # Create individual plots for each metric
    for metric in metrics:
        # Sort classifiers by the metric (descending order)
        df_sorted = df.sort_values(by=metric, ascending=False)

        plt.figure(figsize=(12, 8))
        plt.bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=single_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize the plot
        plt.title(
            f"Classifier Ranking by {metric}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Classifiers", fontsize=14, fontweight="bold")
        plt.ylabel(metric, fontsize=14, fontweight="bold")

        # Set x-axis labels
        plt.xticks(
            range(len(df_sorted)), df_sorted["Classifier"], rotation=45, ha="right"
        )

        # Add value labels on top of bars
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            value = row[metric]
            plt.text(
                i,
                value + 0.01 * max(df_sorted[metric]),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis="y")

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(
            output_dir,
            f"classifier_ranking_{metric.lower().replace('-', '_')}.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved ranking plot for {metric}: {output_path}")

    # Create a comprehensive comparison plot
    plt.figure(figsize=(16, 10))

    # Create grouped bar chart using original scales for all metrics
    x = np.arange(len(df))
    width = 0.13

    for i, metric in enumerate(metrics):
        values = df[metric]
        label = metric

        plt.bar(
            x + i * width,
            values,
            width,
            label=label,
            color=single_color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    plt.title(
        "Classifier Performance Comparison Across All Metrics",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Classifiers", fontsize=14, fontweight="bold")
    plt.ylabel("Metric Values", fontsize=14, fontweight="bold")
    plt.xticks(x + width * 2, df["Classifier"], rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    # Save comprehensive plot
    comprehensive_path = os.path.join(
        output_dir, "classifier_comprehensive_comparison.png"
    )
    plt.savefig(comprehensive_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comprehensive comparison plot: {comprehensive_path}")

    # Create a summary ranking table
    ranking_summary = {}
    for metric in metrics:
        df_ranked = df.sort_values(by=metric, ascending=False)
        ranking_summary[metric] = {
            "Best": df_ranked.iloc[0]["Classifier"],
            "Best_Value": df_ranked.iloc[0][metric],
            "Worst": df_ranked.iloc[-1]["Classifier"],
            "Worst_Value": df_ranked.iloc[-1][metric],
        }

    # Print summary
    print("\n" + "=" * 60)
    print("CLASSIFIER RANKING SUMMARY")
    print("=" * 60)
    for metric, data in ranking_summary.items():
        print(f"\n{metric}:")
        print(f"  Best:  {data['Best']} ({data['Best_Value']:.4f})")
        print(f"  Worst: {data['Worst']} ({data['Worst_Value']:.4f})")

    return ranking_summary
