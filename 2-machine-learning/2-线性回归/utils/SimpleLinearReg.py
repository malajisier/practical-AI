import numpy as np


class SimpleLinearReg1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_pred):
        return np.array([self._predict(x) for x in x_pred])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearReg1()'

# 向量化运算
class SimpleLinearReg2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 转化为数组运算，提高效率
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_pred):
        return np.array([self._predict(x) for x in x_pred])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return 'SimpleLinearReg2()'