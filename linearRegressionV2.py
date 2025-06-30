import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    # for i in range(num_features):
    #     dw[i] = -2/m * sum((y_t - y_p) * X_values[j][i] for j, (y_t, y_p) in enumerate(zip(y_values, y_pred)))
    for i in range(m):
        error = y_values[i] - y_pred[i]
        for j in range(num_features):
            dw[j] += (-2 / m) * X_values[i][j] * error
        db += (-2 / m) * error


    # Compute gradient for bias
    #db = -2/m * sum(y_t - y_p for y_t, y_p in zip(y_values, y_pred))

    return dw, db

def train_linear_regression(X, y, learning_rate, epochs):
    w = [0.1] * X.shape[1]  # Initialize weights - X.shape[1] gives number of columns
    b = 0.0

    for epoch in range(epochs):
        #predict
        y_pred = predict(X, w, b)

        #compute mse
        mse = mean_squared_error(y, y_pred)
        #compute gradients
        dw, db = compute_gradient(X, y, y_pred)

        #update weights and bias
        for i in range(len(w)):
            w[i] -= learning_rate * dw[i]
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, MSE: {mse:.4f}')

    print("Training complete.")

if __name__ == "__main__":
    run_linear_regression()