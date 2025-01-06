import json


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
