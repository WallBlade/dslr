import utils.prepare_data as prep_d
import matplotlib.pyplot as plt
import utils.utils_math as mt
import pandas as pd
import numpy as np
import time
import sys

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def predict(thetas, X, df):
    """
    Show predictions and errors on the dataset used to train the model
    """

    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    errors = 0

    for i, row in df.iterrows():
        prob = np.array([
            hypothesis(thetas[0], X[i]) * 100, # Gryffindor
            hypothesis(thetas[1], X[i]) * 100, # Hufflepuff
            hypothesis(thetas[2], X[i]) * 100, # Ravenclaw
            hypothesis(thetas[3], X[i]) * 100 # Slytherin
        ])
        index = np.argmax(prob)
        if labels[index] == row['Hogwarts House']:
            print(f"\033[32mindex: {i} specie: {row['Hogwarts House']} prediction: {labels[index]} prob: {prob[index]}\033[0m")
        else:
            errors += 1
            print(f"\033[31mindex: {i} specie: {row['Hogwarts House']} prediction: {labels[index]} prob: {prob[index]}\033[0m")

    print(f"Errors: {errors} / {len(df)} ({errors / len(df) * 100:.2f}%)")

def hypothesis(thetas, X):
    z = X@thetas
    return 1 / (1 + np.exp(-z))

def cost_function(thetas, X, y):
    j = np.sum(y * np.log(hypothesis(thetas, X)) + (1 - y) * np.log(1 - hypothesis(thetas, X)))
    return -1 / y.shape[0] * j

def create_mini_batches(X, y, batch_size):
    """
    Create mini-batches from the input data.

    Returns: Array containing list of tuples, where each tuple contains a mini-batch of X and y.
    """

    n_samples = X.shape[0]

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # Shuffle the data
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split into mini-batches
    mini_batches = []
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

def gradient(thetas, X, y):
    m = X.shape[0]
    return (1 / m) * X.T @ (hypothesis(thetas, X) - y)

def plot_cost_history(ax, cost_history, label):
    """
    Plots the cost history on the provided subplot axis.

    Parameters:	- ax: Matplotlib axis object.
                - cost_history: List of cost values over iterations.
                - label: Label for the plot.
    """

    ax.plot(range(len(cost_history)), cost_history, label=label)
    ax.set_title(f"Cost History for {label}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    ax.legend()

def gradient_descent(X, y, ax, label):
    """
    Perform gradient using mini-batch and momentum to find the optimal parameters for logistic regression.
    Use relative change in cost to determine convergence.

    Parameters:	- X: Input data
                - y: Target data ; ax: Matplotlib axis object ; label: Label for the plot.

    Returns: Optimal thetas for logistic regression.
    """

    thetas = np.zeros(X.shape[1])
    velocity = np.zeros(X.shape[1])
    learning_rate = 0.1 
    cost_history = []
    max_epoch = 200
    batch_size = 32
    momentum = 0.9
    i = 0

    for _ in range(max_epoch):
        mini_batches = create_mini_batches(X, y, batch_size)
        for X_mini, y_mini in mini_batches:
            grad = gradient(thetas, X_mini, y_mini)
            velocity = momentum * velocity - learning_rate * grad
            thetas = thetas + velocity
            cost_history.append(cost_function(thetas, X, y))

            if len(cost_history) > 1:
                relative_change = abs(cost_history[-1] - cost_history[-2]) / max(cost_history[-2], 1e-8)
                if relative_change <= 1e-6:
                    print(f"Convergence achieved at iteration {i} with relative cost change {relative_change:.6e}")
                    plot_cost_history(ax, cost_history, label)
                    return thetas
            i += 1

    plot_cost_history(ax, cost_history, label)

    return thetas

def main():
    usage = f'{RED}Usage: python logreg_train.py dataset_train.csv{RESET}'
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    else:
        data = sys.argv[1]
        if data != 'datasets/dataset_train.csv':
            print(usage)
            sys.exit(1)

    try:
        # ---- Get and set up data ---- #
        np.set_printoptions(suppress=True)
        df = prep_d.prepare_data('datasets/dataset_train.csv')
        df.insert(0, 'Bias', 1)

        # ---- Set variables ---- #
        X = df.select_dtypes(include=['number']).values
        r = []
        start_time = time.time()

        # ---- Perform gradient descent for each class ---- #
        fig, axs = plt.subplots(2, 2, figsize=(10, 9))

        i = 0
        for label in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
            y = (df['Hogwarts House'] == label).astype(int).values
            thetas = gradient_descent(X, y, axs[i // 2, i % 2], label)
            r.append(thetas)
            print(f"Thetas {label}: {thetas}", end='\n\n')
            i += 1

        # ---- Print results ---- #
        elapsed_time = time.time() - start_time
        print(f"Temps d'exécution : {elapsed_time:.6f} secondes")
        predict(r, X, df)

        # ---- Write results to file ---- #
        with open('logistic_thetas.txt', 'w') as file:
            for row in r:
                row = map(str, row)
                result = ' '.join(row)
                file.write(result + '\n')

        # ---- Plot graphs ---- #
        plt.show()
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    main()