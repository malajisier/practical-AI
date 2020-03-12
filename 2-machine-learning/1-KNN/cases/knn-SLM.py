import numpy as np
import time

def loadData(path):
    print('loading file...')
    dataArr = []
    labelArr = []

    fr = open(path)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        labelArr.append(int(curLine[0]))
        dataArr.append(int(data) for data in curLine[1:])

    return labelArr, dataArr


def clacDist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))

def knn(trainData, trainLabel, x, topk):
    distances = []
    for i in range(len(trainData)):
        x1 = trainData[i]
        curDist = clacDist(x1, x)
        distances[i] = curDist

    topkList = np.argsort(np.array(distances))[:topk]
    labelList = []
    for index in topkList:
        labelList[int(trainData[index])] += 1

    return labelList.index(max(labelList))

def test(trainData, trainLabel, testData, testLabel, topk):
    trainData = np.mat(trainData)
    trainLabel = np.mat(trainLabel).T
    testData = np.mat(testData)
    testLabel = np.mat(testLabel).T

    errorCount = 0
    for i in range(len(testData)):
        print('test %d: %d' % (i, len(testData)))
        x = testData[i]
        y = knn(trainData, trainLabel, x, topk)
        if y != testLabel[i]:
            errorCount += 1

    return 1 - (errorCount / len(testData))


if __name__ == "__main__":
    start = time.time()

    trainData, trainLabel = loadData()
    testData, testLabel = loadData()

    accuracy = test(trainData, trainLabel, testData, testLabel, 25)
    print('accuracy is: %d' % (accuracy * 100), '%')

    end = time.time()
    print('time cost:', end-start)