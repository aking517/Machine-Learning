import pandas as pd
import random
import math
import operator
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Function that loads our data
def loadDataset(url, split):
    df = pd.read_csv(url, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    # Selects all rows with : and selects every column but the last column :-1
    X = array[:, :-1]
    # Selects the last column -1 which in this case are the answers of the dataset or labels
    y = array[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    return X_train, X_test, y_train, y_test

# Function that gets the accuracy of our tests
def getAccuracy(y_test, y_pred):
    correct = 0
    # Compare Y test with X test to find the accuracy
    for x in range(len(y_test)):
        if y_test[x] == y_pred[x]:
            correct += 1
    return (correct / float(len(y_test))) * 100.0


def main():
    # prepare data
    split = 0.67
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    X_train, X_test, y_train, y_test = loadDataset(url, 0.66)

    # Prepare K range for use in loops
    k_values = range (1,21)
    # Create an array to hold our accuracies
    accuracies = []

    # Loop K for k values 1 through 20
    for k in k_values:
        accuracy_sum = 0
        # Run each K value 5 times and calculate accuracy
        for _ in range(5):
            # Use KNeighborsClassifier function to test our data
            knn_classifier = KNeighborsClassifier(n_neighbors= k)
            knn_classifier.fit(X_train, y_train)
            y_pred = knn_classifier.predict(X_test)
            accuracy_sum += getAccuracy(y_test, y_pred)
        avg_accuracy = accuracy_sum / 5
        accuracies.append(avg_accuracy)
        # Print the accuracy to the user
        print(f'k={k}, average accuracy = {avg_accuracy}%')
    # Create our line chart
    plt.plot(k_values, accuracies, marker = 'o')
    plt.xlabel('k-values')
    plt.ylabel('Accuracy%')
    plt.title('KNN Accuracy comparison')
    plt.grid(True)
    # Display the line chart
    plt.show()

main()
