import numpy as np

class MyKNNClf():
    
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = 0
        self.X_train = None
        self.y_train = None
    
    def __repr__(self):
        return f'MyKNNClf class: k={self.k}, metric={self.metric}, weight={self.weight}'

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.train_size = self.X_train.shape
    
    def _calculate_distance(self, test):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - test) ** 2, axis=1))
        if self.metric == 'chebyshev':
            return np.max(np.abs(self.X_train - test), axis=1)
        if self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - test), axis=1)
        if self.metric == 'cosine':
            return 1 - np.dot(self.X_train, test) / (np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(test))

    def predict(self, X_test):
        X_test = np.array(X_test)
        y_pred = []

        for test in X_test:
            distances = self._calculate_distance(test)
            near_indexes = np.argsort(distances)[:self.k]
            near_classes = self.y_train[near_indexes]
            near_distances = distances[near_indexes]
            
            if self.weight == 'uniform':
                uniq_classes, counts = np.unique(near_classes, return_counts=True)
                pred = uniq_classes[np.argmax(counts)]
            else:
    
                class_weights = {}
                for i, class_label in enumerate(near_classes):
                    if self.weight == 'distance':
                        weight = 1 / (near_distances[i] + 1e-10)  
                    elif self.weight == 'rank':
                        weight = 1 / (i + 1)  
                    
                    if class_label in class_weights:
                        class_weights[class_label] += weight
                    else:
                        class_weights[class_label] = weight
                
                pred = max(class_weights.items(), key=lambda x: x[1])[0]
            
            y_pred.append(pred)
        
        return np.array(y_pred)
    
    def predict_proba(self, X_test):
        X_test = np.array(X_test)
        y_prob = []

        for test in X_test:
            distances = self._calculate_distance(test)
            near_indexes = np.argsort(distances)[:self.k]
            near_classes = self.y_train[near_indexes]
            near_distances = distances[near_indexes]
            
            if self.weight == 'uniform':
                prob = np.mean(near_classes)
            else:
                total_weight = 0
                class1_weight = 0
                
                for i, class_label in enumerate(near_classes):
                    if self.weight == 'distance':
                        weight = 1 / near_distances[i]
                    elif self.weight == 'rank':
                        weight = 1 / (i + 1)
                    
                    total_weight += weight
                    if class_label == 1:
                        class1_weight += weight
                
                prob = class1_weight / total_weight if total_weight > 0 else 0
            
            y_prob.append(prob)

        return np.array(y_prob)

