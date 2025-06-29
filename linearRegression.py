import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


def run_linear_regression():

    print("---- Starting Linear Regression ----")

    print("---- Loading Concrete Compressive Strength Dataset ----")
    try:
        concrete_data = fetch_ucirepo(id=165)
    except Exception as e:
        print("Error loading Concrete Strength Dataset: " + str(e))
        return

    X = concrete_data.data.features
    y = concrete_data.data.targets

    df = pd.concat([X, y], axis=1)

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    feature_x = 'Water'
    feature_y = 'Superplasticizer'

    print("---- Cleaning Dataset if needed ----")
    print("Null values:\n", df.isnull().sum())

    # Check if the chosen features exist in the DataFrame
    if feature_x not in df.columns or feature_y not in df.columns:
        print(f"Error: One or both of the selected features ('{feature_x}', '{feature_y}') not found in the dataset.")
        print("Available columns:", df.columns.tolist())
        exit()

    # Drop rows with NaN values in the selected columns to prevent plotting errors
    df_plot = df[[feature_x, feature_y]].dropna()

    correlation_matrix = df.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True)

    # --- 4. Create the scatterplot ---
    plt.figure(figsize=(10, 7))  # Set the figure size for better readability
    sns.scatterplot(x=feature_x, y=feature_y, data=df_plot, alpha=0.7, edgecolor='w', linewidth=0.5)

    # --- 5. Add titles and labels ---
    plt.title(f'Scatterplot of {feature_x} vs. {feature_y} (UCI Dataset 165)', fontsize=14)
    plt.xlabel(f'{feature_x}', fontsize=12)
    plt.ylabel(f'{feature_y}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add a grid for better readability

    # --- 6. Show the plot ---
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == "__main__":
    run_linear_regression()