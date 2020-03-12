import numpy as np

# 按比例划分训练集与测试集
def train_test_split(X, y, ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal y"
    assert 0.0 <= ratio <= 1.0, "ratio must be valid"

    '''
    permutation 对比 shuffle, 都是用来随机打乱原序列
    permutation: 不是在原数组上进行操作，而是返回一个新的打乱顺序的数组
    shuffle: 在原数组上随机打乱顺序，无返回值
    '''
    if seed:
        np.random.seed(seed)

    shuffled_index = np.random.permutation(len(X))
    test_size = int(len(X) * ratio)
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return X_train, y_train, X_test, y_test



