import utils.prepare_data as prep_d
import utils.utils_math as mt
import pandas as pd
import numpy as np
import sys

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def fill_nan_with_mean(df):
    """
    Fills all NaN values the mean of their respective columns.
    
    Parameters: - df (pd.DataFrame): The input DataFrame
    
    Returns: pd.DataFrame: A new DataFrame with NaN values replaced by column means
    """

    column_means = df.mean(numeric_only=True)
    filled_df = df.fillna(column_means)

    return filled_df

def hypothesis(thetas, X):
    z = X@thetas
    return 1 / (1 + np.exp(-z))

def predict(thetas, X, df):
    """
    Calculate the predictions for each class and returns the highest one (one-vs-all) using logistic regression and print the results.

    Parameters: - thetas (list of lists): The list of theta values for each class.
                - X (numpy.ndarray): The feature matrix.
                - df (pandas.DataFrame): The DataFrame containing the dataset to predict.
    """

    labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    for i, row in df.iterrows():
        prob = np.array([
            hypothesis(thetas[0], X[i]) * 100, # Gryffindor
            hypothesis(thetas[1], X[i]) * 100, # Hufflepuff
            hypothesis(thetas[2], X[i]) * 100, # Ravenclaw
            hypothesis(thetas[3], X[i]) * 100 # Slytherin
        ])
        index = np.argmax(prob)
        print(f"{i},{labels[index]}")

def main():
    if len(sys.argv) != 2 or sys.argv[1] != 'dataset_test.csv':
        print(f'{RED}Usage: python3 logreg_predict.py dataset_test.csv{RESET}')
        sys.exit(1)

    try:
        # ---- Get and set up data ---- #
        np.set_printoptions(suppress=True)
        df = pd.read_csv('datasets/dataset_test.csv')
        df.drop(["Index", "Hogwarts House", "Best Hand", "First Name", "Last Name", "Birthday" ], axis=1, inplace=True)
        df = df.drop(['Arithmancy', 'Defense Against the Dark Arts', 'Care of Magical Creatures'], axis=1)
        df = fill_nan_with_mean(df)
        df = mt.normalize(df)
        df.insert(0, 'Bias', 1)

        X = df.select_dtypes(include=['number']).values
        r = []

        # ---- Get thetas from file ---- #
        with open('logistic_thetas.txt', 'r') as file:
            for line in file:
                thetas = line.strip().split()
                thetas = list(map(float, thetas))
                r.append(thetas)

        # ---- Perform the predictions ---- #
        predict(r, X, df)
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    main()