import pandas as pd
import matplotlib.pyplot as plt
import prepare_data as d
import numpy as np
import sys

def	split_data():
    data = d.prepare_data()
    
    X = data.drop(['Hogwarts House'], axis=1)
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    y = data['Hogwarts House']

    house_mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 2, 'Slytherin': 3}
    y = y.map(house_mapping)

    return X_train, y

def	sigmoid(z):
    return 1 / (1 + np.exp(-z))

def	grad(X, y, thetas, m):
    return 1/m * X.T.dot(np.dot(X, thetas) - y)

def	cost_function(X, y, thetas, m):
    predictions = sigmoid(np.dot(X, thetas))

    return -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

def	gradient_descent(X, y, thetas, learning_rate=0.001, iterations=1000):
    m = len(y)
    cost_history = []
    print(y)

    for _ in range(iterations):
        gradients = grad(X, y, thetas, m)
        thetas -= learning_rate * gradients
        cost_history.append(cost_function(X, y, thetas, m))

    # plt.plot(range(len(cost_history)), cost_history)
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Cost (Log-Loss)')
    # plt.title('Cost Function During Training')
    # plt.show()

    # print(thetas)
    # print(pd.DataFrame(cost_history).to_string())
    return thetas

def	train_model(X, y):
    final_thetas = []

    for i in range(4):
        thetas = np.zeros(X.shape[1])
        y_train_binary = np.where(y == i, 1, 0)
        theta = gradient_descent(X, y_train_binary, thetas)
        final_thetas.append(theta.tolist())
    # print(final_thetas)
    np.savetxt('thetas.csv', final_thetas, delimiter=',', fmt='%.8f')
    return final_thetas
    
def	predict(X, thetas):
    probabilities = []
    
    for theta in thetas:
        probs = sigmoid(np.dot(X, theta))
        probabilities.append(probs)
    
    return np.argmax(probabilities)

def main():
    X, y = split_data()
    
    thetas = train_model(X, y)
    # to_predict = np.array([1, -1.147904, -1.375864, -1.920888, -0.556627, -1.109010, 0.280333, 1.041455, 0.452024, -1.003297, -1.387362])
    # final_thetas = np.loadtxt('thetas.csv')
    # print(thetas)
    # print(final_thetas)
    # predictions = []
    # for student_scores in X:  # X_test is your test dataset
    #     predicted_house = predict(np.array(student_scores, dtype=float), thetas)
    #     predictions.append(predicted_house)

    # print(predictions)
    # proba = predict(to_predict, thetas)
    # print(proba)

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    main()