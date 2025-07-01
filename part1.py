import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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

    # 3. Specify features to exclude
    features_to_exclude = ['Fly Ash']
    X_selected = X[features_to_exclude]
    X_filtered = X.drop(columns=features_to_exclude)


    X_cleaned, Y_cleaned = clean_data(X_filtered, y)
    plot_data(X_cleaned, Y_cleaned)



    # 4. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, Y_cleaned, test_size=0.2, random_state=42)

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    w, b = train_linear_regression(X_train_scaled_df, y_train, 0.01, 1000)
    print("Weights:", w)
    print("Bias:", b)

    # Predict on the test set
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    y_pred_after_train = predict(X_test_scaled_df, w, b)
    y_pred_after_train_df = pd.DataFrame(y_pred_after_train, columns=['Concrete compressive strength'], index=X_test.index)

    mse_after_train = mean_squared_error(y_test, y_pred_after_train_df)
    r2_after_train = r2_score(y_test, y_pred_after_train_df)
    print(f"Mean Squared Error after training: {mse_after_train:.4f}")
    print(f"R^2 Score after training: {r2_after_train:.4f}")







#model =  Y =w1X1 + w2X2 + ... + b
def predict(x, w, b):
    x_values = x.values
    predictions = []
    for row in x_values:
        pred = sum(w[i] * row[i] for i in range(len(w))) + b
        pred = round(pred, 4)
        predictions.append(pred)

    return predictions

def compute_gradient(X, y, y_pred):
    # Convert to numpy arrays for easier computation
    X_values = X.values
    if hasattr(y, 'values'):
        y_values = y.values.flatten()
    else:
        y_values = y

    m = len(y_values)
    num_features = X_values.shape[1]

    # Initialize gradients for weights
    dw = [0.0] * num_features
    db = 0.0
    # Compute gradients for each weight
    for i in range(m):
        error = y_values[i] - y_pred[i]
        for j in range(num_features):
            dw[j] += (-2 / m) * X_values[i][j] * error
        db += (-2 / m) * error


    return dw, db

def train_linear_regression(X, y, learning_rate, epochs):
    w = [0.1] * X.shape[1]  # Initialize weights - X.shape[1] gives number of columns
    b = 0.0

    for epoch in range(epochs):
        #predict
        y_pred = predict(X, w, b)

        #compute mse
        mse = mean_squared_error(y, y_pred)
        r2_lr = r2_score(y, y_pred)
        #compute gradients
        dw, db = compute_gradient(X, y, y_pred)

        #update weights and bias
        for i in range(len(w)):
            w[i] -= learning_rate * dw[i]
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, MSE: {mse:.4f}, R2: {r2_lr:.4f}')

    print("Training complete.")
    return w, b

def plot_data(X,y):
    df = pd.concat([X, y], axis=1)
    print("---- Cleaning Dataset if needed ----")
    print("Null values:\n", df.isnull().sum())

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

if __name__ == "__main__":
    run_linear_regression()