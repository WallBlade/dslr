import utils.prepare_data as prep_d
import numpy as np

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

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
    usage = f'{RED}Usage: python logreg_predict.py dataset_test.csv{RESET}'
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)
    else:
        data = sys.argv[1]
        if data != 'datasets/dataset_test.csv':
            print(usage)
            sys.exit(1)

    try:
        # ---- Get and set up data ---- #
        np.set_printoptions(suppress=True)
        df = prep_d.prepare_data('datasets/dataset_test.csv')
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