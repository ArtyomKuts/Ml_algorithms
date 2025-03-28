import numpy as np
import pandas as pd

class MyKNNReg():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.train_size = None
        self.metric = metric
        self.weight = weight
    
    def __repr__(self):
        return f'MyKNNReg class: k={self.k}, metric={self.metric}'
    
    def _calculate_distance(self, test):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - test) ** 2, axis=1))
        if self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train - test), axis=1)
        if self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - test), axis=1)
        if self.metric == 'cosine':
            return 1 - np.dot(self.X_train, test) / (np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(test))
        
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.train_size = X_train.shape

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = []

        for test in X_test:
            distances = self._calculate_distance(test)
            near_indices = np.argsort(distances)[:self.k]
            near_neighbors = self.y_train[near_indices]
            if self.weight == 'uniform':
                prediction = np.mean(near_neighbors) 
            elif self.weight == 'distance':
                near_distances = distances[near_indices]
                weights = (1 / near_distances) / np.sum(1 / near_distances) 
                prediction = np.sum(weights * near_neighbors)
            elif self.weight == 'rank':
                ranks = np.arange(1, self.k + 1) 
                weights = (1 / ranks) / np.sum(1 / ranks)  
                prediction = np.sum(weights * near_neighbors)
            y_pred.append(prediction)

        return np.array(y_pred)