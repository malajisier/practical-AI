import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z):
    return sigmoid(z)(1 - sigmoid(z))


# class MLP:
#     def __init__(self, sizes):
#         """
#         :param sizes: [784, 30, 10] 每层的神经元数量
#         :out: [batch_size, 30]
#         """
#         self.sizes = sizes
#         # w: [ch_out, ch_in] 倒置方便后续矩阵相乘，使得batch_size在前,使得output=[batch_size, 30]
#         # b: [ch_out]
#         self.weights = [np.random.randint(ch2, ch1) for ch1, ch2 in zip(sizes[:-1], sizes[1:])]  # [784, 30], [30, 10]
#         self.bias = [np.random.randint(ch) for ch in sizes[1:]]

#     def forward(self, x):
#         """
#         :param x: [784, 1]  batch_size = 1
#         :return: [10, 1]
#         """
#         for b, w in zip(self.bias, self.weights):
#             # (30,784)@(784,1) + (30,1) = (30,1)
#             z = np.dot(w, x) + b
#             # 激活后的输出值，作为下一层的输入
#             x = sigmoid(z)

#         return x

#     def backprop(self, x, y):
#         """
#         :param x: [784, 1]
#         :param y: [10, 1] one-hot encoding
#         :return:
#         """
#         # 初始化 w, b
#         del_w = [np.zeros(w.shape) for w in self.weights]
#         del_b = [np.zeros(b.shape) for b in self.bias]

#         """添加前向传播过程的目的是，记录中间变量方便计算"""
#         # 保存每层的激活值，作为下一层的输入
#         activations = [x]
#         zs = []
#         activation = x

#         for b, w in zip(self.bias, self.weights):
#             z = np.dot(w, activation) + b
#             activation = sigmoid(z)

#             zs.append(z)
#             activations.append(activation)

class MLP:
    def __init__(self, sizes):
        self.sizes = sizes
        self.weights = [np.random.randn(ch2, ch1) for ch1, ch2 in zip(sizes[:-1, sizes[1:]])] # [784, 30], [30, 10]
        self.bias = [np.random.randn(b) for b in zip(s)] 



def main():
    pass


if __name__ == '__main__':
    main()
