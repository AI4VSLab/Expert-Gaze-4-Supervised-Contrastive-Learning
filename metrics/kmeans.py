from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.cluster import KMeans
import numpy as np


import numpy as np
from sklearn.neighbors import NearestNeighbors

from collections import Counter

def most_common_element(array):
    # Use Counter to count occurrences of each element
    counts = Counter(array)
    # Find the most common element and its count
    most_common_element, count = counts.most_common(1)[0]
    return most_common_element


def kmeans_clustering(X_train=None, X_test=None, Y_train=None, Y_test=None, n_clusters=2, filenames=None):
    '''
    @returns
        filename2label[filename[i]] = 0/1 which cluster
    
    '''
    # Normalize data across its dimension
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Fit the KMeans model to the training data
    kmeans.fit(X_train)

    # Predict cluster labels for the test set
    test_cluster_labels = kmeans.predict(X_test)

    train_cluster_labels = kmeans.predict(X_train)

    # Generate a binary matrix
    filename2label = {}

    # Fill the binary matrix based on cluster assignments
    for i in range(len(train_cluster_labels)):
        filename2label[filenames[i]] = train_cluster_labels[i]


    # Evaluate clustering performance 
    accuracy = accuracy_score(Y_test, test_cluster_labels)
    print(f'KMeans Test Accuracy: {accuracy:.4f}')

    # Compute the confusion matrix 
    conf_matrix = confusion_matrix(Y_test, test_cluster_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate the Matthews Correlation Coefficient (MCC) 
    mcc = matthews_corrcoef(Y_test, test_cluster_labels)
    print("Matthews Correlation Coefficient (MCC):", mcc)

    return kmeans, accuracy, conf_matrix, mcc, filename2label

