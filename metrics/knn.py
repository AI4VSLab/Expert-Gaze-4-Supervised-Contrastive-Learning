from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef

import numpy as np
from sklearn.neighbors import NearestNeighbors


from collections import Counter

def most_common_element(array):
    # Use Counter to count occurrences of each element
    counts = Counter(array)
    # Find the most common element and its count
    most_common_element, count = counts.most_common(1)[0]
    return most_common_element

def knn(
    X_train = None,
    X_test = None,
    Y_train = None,
    Y_test = None,
    k = 5,
    filenames = None
    ):

    # Normalize data across its dimension -> indeed better
    scaler = StandardScaler()
    X_train = scaler.fit_transform( X_train)
    X_test = scaler.transform( X_test)

    # Initialize the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')

    # Train the KNN classifier
    knn_classifier.fit(X_train, Y_train)

    
    # Predict using the trained KNN classifier
    y_pred = knn_classifier.predict(X_test)

    dist, indices = knn_classifier.kneighbors(X_train)
    num_correct = 0
    result_dict = {}
    for i, neighbor_indices in enumerate(indices):
        if Y_train[neighbor_indices[0]] ==  most_common_element([Y_train[i] for i in neighbor_indices[1:] ]):
            num_correct += 1
        result_dict[filenames[i]] = [filenames[idx] for idx in neighbor_indices[1:]]
    print(f'knn Trainset Accuracy: {num_correct/len(indices)}')

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, y_pred)
    print(f'knn Test Accuracy: {accuracy:.4f}')

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(Y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate the Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(Y_test, y_pred)
    print("Matthews Correlation Coefficient (MCC):", mcc)
    return knn_classifier, accuracy, conf_matrix, mcc, result_dict




def find_closest_neighbors(activation_train, filenames_train, num_neighbors=5):
    # Initialize the NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='auto', metric='minkowski') # cosine
    
    # Fit the model with the activation vectors
    nn_model.fit(activation_train)
    
    # Find the indices of the nearest neighbors for each activation vector
    _, indices = nn_model.kneighbors(activation_train)

    # Create a dictionary to store the results
    result_dict = {}
    # Populate the dictionary with filenames of closest neighbors
    for i, filename in enumerate(filenames_train):
        neighbor_indices = indices[i, 1:]  # Exclude the query point itself
        neighbor_filenames = [filenames_train[idx] for idx in neighbor_indices]
        #if filename in result_dict: print(filename)
        result_dict[filename] = neighbor_filenames
        
    return result_dict