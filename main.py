import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from ucimlrepo import fetch_ucirepo

def run_comprehensive_linear_regression_example():
    """
    This function demonstrates a linear regression example using
    all specified imports, including a real-world dataset,
    scaling, and a comparison with SGDRegressor.
    """

    print("--- Starting Comprehensive Linear Regression Example ---")

    # 1. Load the California Housing Dataset
    print("\n--- Loading Concrete Compressive Strength Dataset ---")
    concrete_strength = fetch_ucirepo(id=165)
    features_df = concrete_strength.data.features
    targets_df = concrete_strength.data.targets
    variables_df = concrete_strength.variables
    X = concrete_strength.data.features
    y = concrete_strength.data.targets

    # Filter this DataFrame to find rows where the 'role' is 'Target'
    target_variables = variables_df[variables_df['role'] == 'Target']

    print("--- Feature Types ---")
    print(features_df.dtypes)

    print("--- Target Types ---")
    print(targets_df.dtypes)

    print("--- Variables Types ---")
    print(variables_df.dtypes)

    print(f"Concrete Compressive Strength X shape: {X.shape}")
    print(f"Concrete Compressive Strength y shape: {y.shape}")
    print(f"Features (X): {X.columns.tolist()}")
    print(f"Target (y): {target_variables['name'].tolist()}")
    print(f"First 5 X values:\n{X.head()}")
    print(f"First 5 y values:\n{y.head()}")

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    # 3. Standardize Features using StandardScaler
    # This is important for many models, especially SGDRegressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames for easier feature access if needed for plotting later
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


    print("\n--- Features Scaled using StandardScaler ---")
    print(f"Mean of first feature (unscaled train): {X_train.iloc[:, 0].mean():.2f}")
    print(f"Mean of first feature (scaled train): {X_train_scaled_df.iloc[:, 0].mean():.2f}")
    print(f"Std Dev of first feature (unscaled train): {X_train.iloc[:, 0].std():.2f}")
    print(f"Std Dev of first feature (scaled train): {X_train_scaled_df.iloc[:, 0].std():.2f}")


    # 4. Initialize and Train the Linear Regression Model (analytical solution)
    print("\n--- Training LinearRegression Model (Analytical) ---")
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_flat)

    print(f"LinearRegression Intercept: {model_lr.intercept_:.2f}")
    print(f"LinearRegression Coefficients (first 5): {model_lr.coef_[:5]}")

    # 5. Make Predictions and Evaluate LinearRegression
    y_pred_lr = model_lr.predict(X_test_scaled)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\n--- LinearRegression Model Evaluation ---")
    print(f"Mean Squared Error (MSE) - LinearRegression: {mse_lr:.2f}")
    print(f"R-squared (R2 Score) - LinearRegression: {r2_lr:.2f}")

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
    y_pred_sgd = model_sgd.predict(X_test_scaled)

    mse_sgd = mean_squared_error(y_test, y_pred_sgd)
    r2_sgd = r2_score(y_test, y_pred_sgd)

    print("\n--- SGDRegressor Model Evaluation ---")
    print(f"Mean Squared Error (MSE) - SGDRegressor: {mse_sgd:.2f}")
    print(f"R-squared (R2 Score) - SGDRegressor: {r2_sgd:.2f}")

    # 8. Visualize the Results for a Single Feature (for illustrative purposes)
    # We'll pick 'MedInc' (Median Income) as it's often strongly correlated with housing price.
    feature_to_plot = 'Cement'
    feature_index = X.columns.get_loc(feature_to_plot)

    # For plotting, we need the original (unscaled) test data for the x-axis
    # and predictions based on the scaled data.
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=y_test_flat, y=y_pred_lr, alpha=0.6, label='LinearRegression Predictions')
    sns.scatterplot(x=y_test_flat, y=y_pred_sgd, alpha=0.6, label='SGDRegressor Predictions', color='orange')
    # To plot the regression line, we need to generate predictions for a range
    # of the chosen feature, keeping other features constant (e.g., at their mean)
    # This is more complex for multi-feature models, so we'll just show the scatter
    # and the overall fit quality through metrics.
    # A proper line visualization for a single feature from a multi-feature model
    # would involve holding other features constant, which is beyond a "simple" plot.
    # Instead, let's just observe the scatter and understand the model's performance from metrics.

    plt.title(f'Actual vs. Predicted Housing Prices (using {feature_to_plot})')
    plt.xlabel(f'{feature_to_plot} (Unscaled)')
    plt.ylabel('Median House Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Another visualization: Residual plot (errors)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.residplot(x=y_pred_lr, y=(y_test_flat - y_pred_lr), color='green', lowess=False, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residuals for LinearRegression')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.residplot(x=y_pred_sgd, y=(y_test_flat - y_pred_sgd), color='purple', lowess=False, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residuals for SGDRegressor')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    print("\n--- Comprehensive Linear Regression Example Finished ---")

if __name__ == "__main__":
    run_comprehensive_linear_regression_example()