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

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    run_sgdregression(X_train, X_test, y_train, y_test)

def load_dataset():
    # 1. Load the California Housing Dataset
    print("\n--- Loading Concrete Compressive Strength Dataset ---")
    concrete_strength = fetch_ucirepo(id=165)
    features_df = concrete_strength.data.features
    targets_df = concrete_strength.data.targets
    variables_df = concrete_strength.variables
    X = concrete_strength.data.features
    y = concrete_strength.data.targets

    features_to_exclude = []
    X_filtered = X.drop(columns=features_to_exclude)

    # Filter this DataFrame to find rows where the 'role' is 'Target'
    target_variables = variables_df[variables_df['role'] == 'Target']

    print("--- Feature Types ---")
    print(features_df.dtypes)

    print("--- Target Types ---")
    print(targets_df.dtypes)

    print(f"Concrete Compressive Strength X shape: {X.shape}")
    print(f"Concrete Compressive Strength y shape: {y.shape}")
    print(f"Features (X): {X.columns.tolist()}")
    print(f"Target (y): {target_variables['name'].tolist()}")
    print(f"First 5 X values:\n{X.head()}")
    print(f"First 5 y values:\n{y.head()}")

    plots(X_filtered, y)

    return X_filtered, y

def run_sgdregression(X_train, X_test, y_train, y_test):
    y_train_flat = y_train.values.ravel()

    # 3. Standardize Features using StandardScaler
    # This is important for many models, especially SGDRegressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Initialize and Train the SGDRegressor Model (Stochastic Gradient Descent)
    # Note: SGDRegressor typically requires more tuning (learning rate, epochs, etc.)
    # and might not perform as well as LinearRegression out-of-the-box on simple problems.
    print("\n--- Training SGDRegressor Model (Stochastic Gradient Descent) ---")
    # For a simple example, we'll use default parameters, but in practice, you'd tune them.
    # `max_iter` is the number of passes over the training data.
    # `early_stopping=True` helps prevent overfitting by stopping training when validation score doesn't improve.
    model_sgd = SGDRegressor(max_iter=10000, tol=1e-4, random_state=42, early_stopping=True)
    model_sgd.fit(X_train_scaled, y_train_flat)

    print(f"SGDRegressor Intercept: {model_sgd.intercept_[0]:.2f}")
    print(f"SGDRegressor Coefficients (first 5): {model_sgd.coef_[:5]}")

    # 7. Make Predictions and Evaluate SGDRegressor
    y_train_predict = model_sgd.predict(X_train_scaled)

    train_mse_sgd = mean_squared_error(y_train, y_train_predict)
    train_r2_sgd = r2_score(y_train, y_train_predict)

    print("\n--- SGDRegressor Model Training Evaluation ---")
    print(f"Mean Squared Error (MSE) - SGDRegressor: {train_mse_sgd:.2f}")
    print(f"R-squared (R2 Score) - SGDRegressor: {train_r2_sgd:.2f}")

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

    feature_x = 'Water'
    feature_y = 'Superplasticizer'

    # Drop rows with NaN values in the selected columns to prevent plotting errors
    df_plot = df[[feature_x, feature_y]].dropna()

    correlation_matrix = df.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True)

    # Create the scatterplot ---
    plt.figure(figsize=(10, 7))  # Set the figure size for better readability
    sns.scatterplot(x=feature_x, y=feature_y, data=df_plot, alpha=0.7, edgecolor='w', linewidth=0.5)

    # Add titles and labels ---
    plt.title(f'Scatterplot of {feature_x} vs. {feature_y} (UCI Dataset 165)', fontsize=14)
    plt.xlabel(f'{feature_x}', fontsize=12)
    plt.ylabel(f'{feature_y}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add a grid for better readability

    # Show the plot ---
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

if __name__ == "__main__":
    part_two()