import numpy as np
import pandas as pd
import random

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, best_score=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = best_score
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def _calculate_metric(self, y, y_pred):
        error = y_pred - y

        if self.metric == 'mse':
            return np.mean(error ** 2)
        elif self.metric == 'mae':
            return np.mean(abs(error))
        elif self.metric == 'rmse':
            return np.sqrt(np.mean(error ** 2))
        elif self.metric == 'r2':
            return 1 - np.sum((error ** 2)) / np.sum((y - np.mean(y)) ** 2)
        elif self.metric == 'mape':
            return 100 * np.mean(np.abs(error / y))
        
    
    def fit(self, x, y, verbose=False):
        random.seed(self.random_state)
        x = np.c_[np.ones(x.shape[0]), x]
        y = np.array(y)

        if self.weights is None:
            self.weights = np.ones(x.shape[1])
        
        for i in range(1, self.n_iter+1):
            
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float) and 0 < self.sgd_sample < 1:
                    sample_size = int(self.sgd_sample * x.shape[0])
                    sample_rows_idx = random.sample(range(x.shape[0]), sample_size)
                    x_batch = x[sample_rows_idx]
                    y_batch = y[sample_rows_idx]
                elif isinstance(self.sgd_sample, int):
                    sample_size = min(self.sgd_sample, x.shape[0])
                    sample_rows_idx = random.sample(range(x.shape[0]), sample_size)
                    x_batch = x[sample_rows_idx]
                    y_batch = y[sample_rows_idx]
            else:
                x_batch = x
                y_batch = y

            if callable(self.learning_rate):
                current_learning_rate = self.learning_rate(i)
            else:
                current_learning_rate = self.learning_rate

            y_pred = x_batch @ self.weights
            error = y_pred - y_batch
            mse = np.mean(error ** 2)

            if self.reg == 'l1':
                loss = self.l1_coef * abs(self.weights)
            elif self.reg == 'l2':
                loss = self.l2_coef * self.weights ** 2
            elif self.reg == 'elasticnet':
                loss = self.l1_coef * abs(self.weights) + self.l2_coef * self.weights ** 2

            if verbose and i % verbose == 0:
                y_pred_full = x @ self.weights
                metric_value = self._calculate_metric(y, y_pred_full)
                if i == 0:
                    print(f'start | loss:{mse + loss} | {self.metric}:{metric_value}')
                else:
                    print(f'{i} | loss:{mse + loss} | {self.metric}:{metric_value}')

            if self.reg == 'l1':
                gradient = 2 / x_batch.shape[0] * (x_batch.T @ error) + self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                gradient = 2 / x_batch.shape[0] * (x_batch.T @ error) + self.l2_coef * 2 * self.weights
            elif self.reg == 'elasticnet':
                gradient = 2 / x_batch.shape[0] * (x_batch.T @ error) + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
            else:
                gradient = 2 / x_batch.shape[0] * (x_batch.T @ error)

            self.weights -= current_learning_rate * gradient

        if self.metric is not None:
            y_pred_full = x @ self.weights
            metric_value = self._calculate_metric(y, y_pred_full)
            self.best_score = metric_value 

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        return x @ self.weights
                         
    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def get_coef(self):
        return np.mean(self.weights[1:])
    
    def get_best_score(self):
        return self.best_score