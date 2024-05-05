from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

dataset = load_digits()

X = dataset.data/16
y = dataset.target

no_of_iterations = 30
learning_rate = 0.001
split= 1000

x_train = X[:split]
x_test = X[split:]
y_train = y[:split]
y_test = y[split:]
y_train_list = []
y_test_list = []

for i in y_train:
    tmp_list = []
    for j in range(10):

        if i == j:
            tmp_list.append(1)
        else:
            tmp_list.append(0)
    y_train_list.append(tmp_list)

for i in y_test:
    tmp_list = []
    for j in range(10):

        if i == j:
            tmp_list.append(1)
        else:
            tmp_list.append(0)
    y_test_list.append(tmp_list)

y_train = np.array(y_train_list)
y_test = np.array(y_test_list)

weights = np.zeros(640).reshape(10,64)
bias = np.zeros(10).reshape(10,1)

for n in range(no_of_iterations):
    weights_sum = np.zeros(640).reshape(10, 64)
    bias_sum = np.zeros(10).reshape(10, 1)
    score = 0
    for i in range(len(x_train)):
        row = x_train[i]
        x = np.array(row).reshape(64, 1)
        predicted = np.dot(weights, x) + bias
        predicted_number = np.argmax(predicted)
        actual = np.array(y_train[i]).reshape(10, 1)
        actual_number = np.argmax(actual)

        if predicted_number == 0:
            err = actual - np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 1:
            err = actual - np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 2:
            err = actual - np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 3:
            err = actual - np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 4:
            err = actual - np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 5:
            err = actual - np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 6:
            err = actual - np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).reshape(10, 1)
        elif predicted_number == 7:
            err = actual - np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(10, 1)
        elif predicted_number == 8:
            err = actual - np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).reshape(10, 1)
        elif predicted_number == 9:
            err = actual - np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(10, 1)

        if predicted_number == actual_number:
            score += 1

        accuracy = (score / (i + 1)) * 100
        print(f"Training.... Iteration: {n} Accuracy: {accuracy}")

        for j in range(10):
            for k in range(64):
                weights_sum[j][k] += err[j] * x[k]
            bias_sum[j][0] += err[j]
    md = (-2 / len(x_train)) * weights_sum
    bd = (-2 / len(x_train)) * bias_sum
    weights = weights - learning_rate * md
    bias = bias - learning_rate * bd

score = 0
for i in range(len(x_test)):
    row = x_test[i]
    x = np.array(row).reshape(64, 1)
    predicted = np.dot(weights, x) + bias
    predicted_number = np.argmax(predicted)
    actual = np.array(y_test[i]).reshape(10, 1)
    actual_number = np.argmax(actual)

    if predicted_number == actual_number:
        score += 1

    accuracy = (score / (i + 1)) * 100
    print(f"Testing....Actual number: {actual_number} Predicted_number: {predicted_number} Accuracy: {accuracy}")