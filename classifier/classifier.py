import numpy as np

from classifier.utils import count_max_norm

class LinearBinaryClassifier:
    def __init__(self, X, y, mi=0.5, w=[0, 0], b=0):
        self.X = X
        self.y = y
        self.mi = mi
        self.w = np.array(w)
        self.b = b
        self.r = count_max_norm(self.X)

        self.num_of_points = X.shape[0]
        self.accuracies = [self.accuracy()]
        self.b_values = [self.b]
        self.w_values = [self.w]
        self.iterations = 0

    def stop(self):
        for xi, yi in zip(self.X, self.y):
            if np.sign(self.w.dot(xi) - self.b) != yi:
                return False
        return True

    def train(self):
        """
        Algorithm of training is from
        L.Hamel "Knowledge discovery with support vector machines"
        """
        while not self.stop():
            for xi, yi in zip(self.X, self.y):
                if np.sign(self.w.dot(xi) - self.b) != yi: 
                    self.w = self.w + self.mi * yi * xi
                    self.b = self.b - self.mi * yi * self.r ** 2

                self.iterations += 1
                accuracy = self.accuracy()
                self.accuracies.append(accuracy)
                self.w_values.append(self.w)
                self.b_values.append(self.b)
                if accuracy == 1:
                    break

    def accuracy(self):
        goods = 0
        for xi, yi in zip(self.X, self.y):
            if np.sign(self.w.dot(xi) - self.b) == yi: 
                goods += 1
        return goods / self.num_of_points

    def get_iterations(self):
        return self.iterations

    def get_accuracies(self):
        return self.accuracies

    def get_b_values(self):
        return np.array(self.b_values)

    def get_w_values(self):
        return np.array(self.w_values)
