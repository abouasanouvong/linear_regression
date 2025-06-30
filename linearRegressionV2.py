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

    plot_data(X, y)

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    train_linear_regression(X_train_scaled_df, y_train, 0.01, 1000)


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

if __name__ == "__main__":
    run_linear_regression()