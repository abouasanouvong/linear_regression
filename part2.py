import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from ucimlrepo import fetch_ucirepo

def part_two():
    X, y = load_dataset()

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    run_sgdregression(X_train, X_test, y_train, y_test)

def load_dataset():
    # Loading the Dataset
    print("\n--- Loading Concrete Compressive Strength Dataset ---")
    try:
        concrete_data = fetch_ucirepo(id=165)
    except Exception as e:
        print("Error loading Concrete Strength Dataset: " + str(e))
        return None

    X, y = clean_data(concrete_data.data.features, concrete_data.data.targets)

    plots(X, y)

    return X, y

def clean_data(X, y):
    print("\n--- Cleaning Data ---")

    # Variables to be excluded for optimization
    features_to_exclude = ['Coarse Aggregate', 'Fine Aggregate']
    X_filtered = X.drop(columns=features_to_exclude)

    # Combine X_filtered(features) and y(target) temporarily for unified cleaning based on index
    combined_df = pd.concat([X_filtered, y], axis=1)

    print("--- Original X and y Shapes ---")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\n--- Initial Combined DataFrame Info ---")
    combined_df.info()

    # Handling Null Values
    print("\n--- Checking for null values in the dataset, if present remove ---")
    null_values = combined_df.isnull().sum()
    if null_values.sum() > 0:
        print("\n--- Dropping null Values ---")
        combined_df.dropna(inplace=True)

    # Handling Outliers using IQR (Winsorization/Capping)
    print("\n--- Outlier Handling (Winsorization using IQR method) ---")

    # Features to apply outlier handling
    features_to_process = X_filtered.columns

    # Calculate Q1, Q3, and IQR for each numerical column within these specific features
    Q1 = combined_df[features_to_process].quantile(0.25)
    Q3 = combined_df[features_to_process].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply Winsorization to the X part of combined_df
    for column in features_to_process:  # Iterate through the feature column names
        # Cap values below the lower bound for the current column
        combined_df[column] = np.where(
            combined_df[column] < lower_bound[column],  # Access the specific bound using column name
            lower_bound[column],
            combined_df[column]
        )
        # Cap values above the upper bound for the current column
        combined_df[column] = np.where(
            combined_df[column] > upper_bound[column],
            upper_bound[column],
            combined_df[column]
        )

    print("\n--- Combined DataFrame after Winsorization (Outlier Capping) ---")
    print(combined_df[features_to_process].describe())

    # Checking for Duplicates in the data
    print("\n--- Duplicate Rows Check ---")
    duplicate_data = combined_df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_data}")

    if duplicate_data > 0:
        print(f"Removing {duplicate_data} duplicate rows from combined_df...")
        combined_df_cleaned = combined_df.drop_duplicates()
        print(f"Shape after removing duplicates: {combined_df_cleaned.shape}")
    else:
        combined_df_cleaned = combined_df.copy()
        print("No duplicate data found.")

    # Separate X and y again from the cleaned combined_df
    X_cleaned = combined_df_cleaned[X_filtered.columns]
    y_cleaned = combined_df_cleaned[y.columns]

    # --- Final Cleaned DataFrames ---
    print("\n--- Cleaned Features Info ---")
    print("\t--- Information Extraction ---")
    X_cleaned.info()
    print(f"\t--- Descriptive Statistics ---\n{X_cleaned.describe()}")

    print("\n--- Cleaned Target Info ---")
    print("\t--- Information Extraction ---")
    y_cleaned.info()
    print(f"\t--- Descriptive Statistics ---\n{y_cleaned.describe()}")

    return X_cleaned, y_cleaned

def run_sgdregression(X_train, X_test, y_train, y_test):
    y_train_flat = y_train.values.ravel()

    # Standardize Features using StandardScaler
    # This is important for many models, especially SGDRegressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and Train the SGDRegressor Model (Stochastic Gradient Descent)
    print("\n--- Training SGDRegressor Model (Stochastic Gradient Descent) ---")
    model_sgd = SGDRegressor(max_iter=10000, tol=1e-4, random_state=42, early_stopping=True)
    model_sgd.fit(X_train_scaled, y_train_flat)

    print(f"SGDRegressor Intercept: {model_sgd.intercept_[0]:.2f}")
    print(f"SGDRegressor Coefficients (first 5): {model_sgd.coef_[:5]}")

    # Make Predictions and Evaluate SGDRegressor on training data
    y_train_predict = model_sgd.predict(X_train_scaled)

    train_mse_sgd = mean_squared_error(y_train, y_train_predict)
    train_r2_sgd = r2_score(y_train, y_train_predict)

    print("\n--- SGDRegressor Model Training Evaluation ---")
    print(f"Mean Squared Error (MSE) - SGDRegressor: {train_mse_sgd:.2f}")
    print(f"R-squared (R2 Score) - SGDRegressor: {train_r2_sgd:.2f}")

    # Make Predictions and Evaluate SGDRegressor on test data
    y_test_predict = model_sgd.predict(X_test_scaled)

    test_mse_sgd = mean_squared_error(y_test, y_test_predict)
    test_r2_sgd = r2_score(y_test, y_test_predict)

    print("\n--- SGDRegressor Model Testing Evaluation ---")
    print(f"Mean Squared Error (MSE) - SGDRegressor: {test_mse_sgd:.2f}")
    print(f"R-squared (R2 Score) - SGDRegressor: {test_r2_sgd:.2f}")

def plots(X, y):
    df = pd.concat([X, y], axis=1)

    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    # Correlation matrix
    correlation_matrix = df.corr().round(2)

    sns.heatmap(data=correlation_matrix, annot=True)

    # A list of tuples for top 5 correlations
    top_corr = [
        ("Superplasticizer", "Water"),
        ("Superplasticizer", "Fly Ash"),
        ("Fly Ash", "Cement"),
        ("Blast Furnace Slag", "Fly Ash"),
        ("Fly Ash", "Water")
    ]

    for i in range(5):
        feature_x, feature_y = top_corr[i]

        # Create the scatterplot ---
        plt.figure(figsize=(10, 7))  # Set the figure size for better readability
        sns.scatterplot(x=feature_x, y=feature_y, data=df, alpha=0.7, edgecolor='w', linewidth=0.5)
        sns.scatterplot(data=df, x=feature_x, y=feature_x,
                        size="Concrete compressive strength", hue="Concrete compressive strength",
                        palette="viridis", alpha=0.5)

        # Add titles and labels ---
        plt.title(f'Scatterplot of {feature_x} vs. {feature_y} (UCI Dataset 165)', fontsize=14)
        plt.legend(title="Concrete compressive strength", bbox_to_anchor=(1.05, 0.95), loc="upper left")
        plt.xlabel(f'{feature_x}', fontsize=12)
        plt.ylabel(f'{feature_y}', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)  # Add a grid for better readability

        # Show the plot ---
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()

if __name__ == "__main__":
    part_two()