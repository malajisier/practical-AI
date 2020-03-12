from numpy import *
import numpy as np
from collections import Counter
import operator


def createDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inx, dataSet, labels, k):
    # -------------------   方法一：    -------------------------------------
    # numpy broadcasting，axis=1矩阵每一行相加
    distances = sqrt(np.sum((inx - dataSet) ** 2, axis=1))

    # 返回升序排序后结果的索引
    sorted = distances.argsort()[0: k]
    # 按索引取出label
    k_labels = [labels[idx] for idx in sorted]
    # Counter统计标签出现次数, most_common返回出现次数最多的标签tuple
    label = Counter(k_labels).most_common(1)[0][0]
    return label

    # -----------------------   方法二：    --------------------------------
    # dataSetSize = dataSet.shape[0]
    # # tile: 复制行列
    # # 计算inX到每个点的距离
    # diffMat = tile(inx, (dataSet, 1)) - dataSet
    # dist = sqrt(sum(diffMat ** 2))
    # sorted_dist = dist.argsort()
    #
    # classCount = {}
    # for i in range(k):
    #     # 找到该样本的label
    #     voteLabel = labels[sorted_dist[i]]
    #     """
    #     字典的get方法：
    #     如:dict.get(k,d), 参数k在字典中返回k对应的value，k不在字典则返回参数d
    #     """
    #     # 在字典中将该类型+1
    #     classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    #
    # # 返回字典中value最大的key
    # maxClassCount = max(classCount, key=classCount.get)
    # return maxClassCount


def test1():
    group, labels = createDataset()
    print(str(group))
    print(str(labels))
    print(classify([0.1, 0.1], group, labels, 3))


def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据行的行数
    rows = len(fr.readlines())
    # 生成对应的空矩阵
    returnMat = zeros((rows, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return

    index = 0
    for line in fr.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到
    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
    # 计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 极差
    ranges = maxVals - minVals
    # -------第一种实现方式---start-------------------------
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide

    # # -------第二种实现方式---start---------------------------------------
    # norm_dataset = (dataset - minvalue) / ranges

    return normDataSet, ranges, minVals


def datingClassTest():
    """
    对约会网站的测试方法
    :return: 错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


if __name__ == '__main()__':
    # test1()
    datingClassTest()
