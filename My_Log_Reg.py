import numpy as np
import pandas as pd

class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _log_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) 
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def __repr__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def _accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    def _precision(self, y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives 
    
    def _recall(self, y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_true == 1)
        return true_positives / predicted_positives
    
    def _f1(self, y_true, y_pred):
        precision = self._precision(y_true, y_pred)
        recall = self._recall(y_true, y_pred)
        return 2 * precision * recall / (precision + recall)
    
    def _calculate_metric(self, y_true, y_pred):
        if self.metric == 'accuracy':
            return self._accuracy(y_true, y_pred)
        elif self.metric == 'precision':
            return self._precision(y_true, y_pred)
        elif self.metric == 'recall':
            return self._recall(y_true, y_pred)
        elif self.metric == 'f1':
            return self._f1(y_true, y_pred)
        
    
    def fit(self, x, y, verbose=False):
        x = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.ones(x.shape[1])

        for i in range(self.n_iter):
            y_pred = self._sigmoid(x @ self.weights)
            logloss = self._log_loss(y, y_pred)

            if self.metric:
                if self.metric == 'accuracy':
                    metric_value = self._accuracy(y, (y_pred > 0.5).astype(int))
                elif self.metric == 'precision':
                    metric_value = self._precision(y, (y_pred > 0.5).astype(int))
                elif self.metric == 'recall':
                    metric_value = self._recall(y, (y_pred > 0.5).astype(int))
                elif self.metric == 'f1':
                    metric_value = self._f1(y, (y_pred > 0.5).astype(int))
                elif self.metric == 'roc_auc':
                    metric_value = self._roc_auc(y, y_pred)

            if self.reg == 'l1':
                loss = self.l1_coef * abs(self.weights)
            elif self.reg == 'l2':
                loss = self.l2_coef * self.weights ** 2
            elif self.reg == 'elasticnet':
                loss = self.l1_coef * abs(self.weights) + self.l2_coef * self.weights ** 2

            if verbose and i % verbose == 0:
                if i == 0:
                    print(f'start | loss:{logloss + loss} | {self.metric}:{metric_value}')
                else:
                    print(f'{i} | loss:{logloss + loss} | {self.metric}:{metric_value}')

            if self.reg == 'l1':
                gradient = ((y_pred - y) @ x) / y.size + self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                gradient = ((y_pred - y) @ x) / y.size + self.l2_coef * 2 * self.weights
            elif self.reg == 'elasticnet':
                gradient = ((y_pred - y) @ x) / y.size + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            else:
                gradient = ((y_pred - y) @ x) / y.size

            self.weights -= self.learning_rate * gradient
            gradient = ((y_pred - y) @ x) / y.size
            self.weights -= self.learning_rate * gradient

            if self.metric:
                self.best_score = metric_value
    
    def predict_proba(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        return self._sigmoid(x @ self.weights)
    
    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype(int)
    
    def get_coef(self):
        return self.weights[1:]
    
    def get_best_score(self):
        return self.best_score