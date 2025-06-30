import pandas as pd
from sklearn.model_selection import train_test_split
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

    # 2. Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Data Split ---")
    print(f"Training data size: {len(X_train)} samples")
    print(f"Testing data size: {len(X_test)} samples")

    # Train and get MSE history for plotting
    X, Y = generate_multi_feature_data()
    #w, b, mse_history = train_linear_regression(X_train, y_train, 0.01, 1000)
    w, b, mse_history = train_linear_regression(X, Y, 0.01, 1000)

    # Plot the learning curve
    plot_learning_curve(mse_history)

    # Test the model
    # y_test_pred = predict(X_test, w, b)
    # test_mse = compute_mse(y_test, y_test_pred)
    # print(f"\nTest MSE: {test_mse:.4f}")

def plot_learning_curve(mse_history):
    """Plot the learning curve showing how MSE decreases over epochs"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(mse_history, 'b-', linewidth=2)
    plt.title('Learning Curve - MSE vs Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print learning statistics
    initial_mse = mse_history[0]
    final_mse = mse_history[-1]
    improvement = ((initial_mse - final_mse) / initial_mse) * 100

    print(f"\nLearning Statistics:")
    print(f"Initial MSE: {initial_mse:.6f}")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Improvement: {improvement:.2f}%")

#model =  Y =w1X1 + w2X2 + ... + b
def predict(X, w, b):
    predictions = [sum(w[i] * x[i] for i in range(len(x))) + b for x in X]

    return predictions

#mse = mean squared error
def compute_mse(y_true, y_pred):

    if hasattr(y_true, 'values'):
        y_true = y_true.values.flatten()

    # Debug: Check for NaN values
    if any(pd.isna(y_true)):
        print("Warning: NaN values found in y_true")
    if any(pd.isna(y_pred)):
        print("Warning: NaN values found in y_pred")

    N = len(y_true)
    mse =  sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)) / N

    # Debug: Check if MSE is NaN
    if pd.isna(mse):
        print("Warning: MSE calculation resulted in NaN")
        print(f"Sample y_true values: {y_true[:5]}")
        print(f"Sample y_pred values: {y_pred[:5]}")

    return mse

def compute_gradient(X, y, y_pred):
    # Convert to numpy arrays for easier computation
    #X_values = X.values
    #if hasattr(y, 'values'):
       # y_values = y.values.flatten()
    #else:
        #y_values = y

    N = len(y)
    num_features = len(X[0])

    # Initialize gradients for weights
    dw = [0.0] * num_features
    db = 0.0  # Gradient for bias

    # Compute gradients for each weight
    for i in range(N):
        error = y[i] - y_pred[i]
        for j in range(num_features):
            dw[j] += (-2 / N) * X[i][j] * error
        db += (-2 / N) * error


    return dw, db

def train_linear_regression(X, y, learning_rate, epochs):

    #w = [0.0] * X.shape[1]  # Initialize weights - X.shape[1] gives number of columns
    w = [0.0] * len(X[0])
    b = 0.0
    mse_history = []  # Store MSE values for plotting

    for epoch in range(epochs):
        #predict
        y_pred = predict(X, w, b)

        #compute mse
        mse = compute_mse(y, y_pred)
        mse_history.append(mse)  # Store MSE for this epoch

        #compute gradients
        dw, db = compute_gradient(X, y, y_pred)

        #update weights and bias
        for i in range(len(w)):
            w[i] -= learning_rate * dw[i]
        b -= learning_rate * db

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, MSE: {mse:.4f}')

    print("Training complete.")
    print(f"Final MSE: {mse_history[-1]:.4f}")

    return w, b, mse_history  # Return weights, bias, and MSE history


def generate_multi_feature_data():
    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ]
    Y = [
        2 * 1 + 3 * 2 + 5 + 0.1,  # ≈ 13.1
        2 * 2 + 3 * 3 + 5 - 0.2,  # ≈ 15.8
        2 * 3 + 3 * 4 + 5 + 0.3,  # ≈ 18.3
        2 * 4 + 3 * 5 + 5 - 0.1,  # ≈ 20.9
        2 * 5 + 3 * 6 + 5 + 0.2  # ≈ 23.2
    ]
    return X, Y


if __name__ == "__main__":
    run_linear_regression()