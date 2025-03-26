import numpy as np
import pandas as pd
import random

class MySVM():
    
    def __init__(self, n_iter=10, learning_rate=0.001, weights=None, b=None, C=1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def _calculate_loss(self, x, y):
        hingle_loss = 0
        for i in range(len(x)):
            hingle_loss += np.linalg.norm(self.weights) ** 2 + self.C * max(0, 1 - y[i] * (self.weights @ x[i] + self.b)) / y.shape[0]
        hingle_loss /= len(x)
        return np.linalg.norm(self.weights) ** 2 + hingle_loss

    def fit(self, x, y, verbose=False):
        random.seed(self.random_state)
        y = y.mask(y == 0, -1)
        y = np.array(y)
        x = np.array(x)
        self.weights = np.ones(x.shape[1])
        self.b = 1

        for i in range(1, self.n_iter + 1):
            loss = self._calculate_loss(x, y)
            if verbose and i % verbose == 0:
                if i == 0:
                    print(f'start | loss: {loss}')
                else:
                    print(f'{i} | loss: {loss}')
            
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

            for j in range(y_batch.shape[0]):
                if y_batch[j] * (self.weights @ x_batch[j] + self.b) >= 1:
                    gradient_w = 2 * self.weights
                    gradient_b = 0
                else:
                    gradient_w = 2 * self.weights - self.C * y_batch[j] * x_batch[j]
                    gradient_b = - self.C * y_batch[j]
                
                self.weights -= self.learning_rate * gradient_w
                self.b -= self.learning_rate * gradient_b
    
    def predict(self, x):
        x = np.array(x)
        y = np.zeros(x.shape[0], dtype=int)
        for i in range(x.shape[0]):
            y[i] += np.sign(self.weights @ x[i] + self.b)
        y = np.where(y == -1, 0, y)
        return y
            

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def get_coef(self):
        return (self.weights, self.b)