#!/usr/bin/python3
# -*-coding:utf-8-*-

# **********************************************************
# * Author        : SnailCL
# * Create time   : 2020-08-25 13:14:02
# * Last modified : 2020-08-25 14:38:19
# * Filename      : kNN.py
# * Description   :
# **********************************************************

import matplotlib.pyplot as plt
import numpy as np
import operator
import os


def createDatSet():
    """
    Description: 创建数据集和标签
    Parameters: None
    Returns: group-数据集 ndarray
             labels-标签   list
    """
    # 创建二维数组
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 创建标签
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, datSet, labels, k):
    """
    Description: 对数据集进行分类
    Paremeters: inX-特征输入向量 list
                datSet-数据集 ndarray
                labels-标签 list
                k-前k个最相似的数据 int
    Returns: sortedClassCount[0][0] 类型最多的分类
    """
    # 计算datSet的第一维度（行向量）个数
    datSetSize = datSet.shape[0]
    # 使用欧式距离公式求目标点与数据集中每一个点的距离
    # d = [(Ax - Bx) ** 2 + (Ay - By) ** 2] ** 0.5
    # np.tile 构造与数据集同型的数组, 并作差
    diffMat = np.tile(inX, (datSetSize, 1)) - datSet
    sqDiffMat = diffMat ** 2             # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 求和(横向)
    distances = sqDistances ** 0.5       # 开方
    # 得到distances从小到大排序后的索引值
    sortedDistIndicies = distances.argsort()
    # 构建记录类别的字典
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 累加分类次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对记录字典进行从大到小排序，排序依据是该类别的次数
    sortedClassCount = sorted(
        classCount.items(),
        key=operator.itemgetter(1),  # 创建一个获取第1维数据的函数
        reverse=True
    )
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    Deprecation: 将文件中的数据构建成数组
    Parameters: filename-文件路径 str
    Returns: returnMat-特征矩阵 ndarray
             classLabelVector-分类向量 list
    """
    # 打开文件
    fp = open(filename)
    # 按行读取所有的值
    listAllLines = fp.readlines()
    # 获取文件行数
    numberOfLines = len(listAllLines)
    # 构建特征矩阵, n行3列的矩阵
    returnMat = np.zeros((numberOfLines, 3))
    # 构建返回的标签列表
    classLabelVector = []
    index = 0
    for line in listAllLines:
        line = line.strip()                     # 去掉两头的空白符
        listFromLine = line.split('\t')         # 按制表符划分成列表
        returnMat[index, :] = listFromLine[:3]  # 将前三个值赋给特征矩阵的第index行
        classLabelVector.append(int(listFromLine[-1]))  # 将最后一个值赋给标签列表的index位置
        index += 1
    # 关闭文件
    fp.close()
    return returnMat, classLabelVector


def autoNorm(datSet):
    """
    Description: 数据归一化处理
    Parameters: datSet-原始数据 ndarray
    Returns: normDatSet-归一化后的数据集
             ranges-范围 ndarray
             minVals-最小值 ndarray
    """
    # 获取列最大值，最小值
    minVals = datSet.min(0)
    maxVals = datSet.max(0)
    ranges = maxVals - minVals  # 范围
    # 构建数据集同型数组
    normDatSet = np.zeros(np.shape(datSet))
    # 获取第一维度值
    rows = datSet.shape[0]
    normDatSet = datSet - np.tile(minVals, (rows, 1))     # 减去最小值
    normDatSet = normDatSet / np.tile(ranges, (rows, 1))  # 除以范围, 完成归一化
    return normDatSet, ranges, minVals


def datingClassTest(filename):
    """
    Description: 验证分类函数的错误率
    Parameters: filename-文件路径 str
    Returns: None
    """
    hoRatio = 0.10  # 测试集比例
    # 读取文本数据
    datingMat, datingLabels = file2matrix(filename)
    # 数据归一化
    normMat, ranges, minVals = autoNorm(datingMat)
    # 切分数据集
    m = normMat.shape[0]  # 获取特征向量的个数
    numTestVecs = int(m * hoRatio)  # 测试特征个数
    errorCount = 0.0
    # 默认前numTestVecs个数据为测试集，numTestVecs-m个数据为训练集
    for i in range(numTestVecs):
        # 测试数据，训练集，训练集对应的标签，k值
        classifierRes = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(f'the classifier came back with: {classifierRes}, the real answer is {datingLabels[i]}')
        if classifierRes != datingLabels[i]:
            errorCount += 1
    print(f'the total error rate is: {errorCount / float(numTestVecs)}')


def classifyPerson(filename):
    """
    Description: 根据输入的数据预测结果
    Parameters: filename-文件路径 str
    Returns: None
    """
    listRes = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    try:
        q1 = float(input('每年飞行里程数：'))
        q2 = float(input('玩视频游戏所消耗的时间百分比：'))
        q3 = float(input('每周消耗冰淇淋公升数：'))
    except Exception:
        return True
    inArray = np.array([q1, q2, q3])
    datingMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingMat)
    normInArray = (inArray - minVals) / ranges
    classifierRes = classify0(normInArray, normMat, datingLabels, 3)
    print(f'对你而言这是一个{listRes[classifierRes - 1]}')


def img2vector(filename):
    """
    Description: 将图像文件转为数组
    Parameters: filename-文件路径 str
    Returns: returnVect-数组 ndarray
    """
    # 构建returnVect的型: 二维，1行1024列
    returnVect = np.zeros((1, 1024))
    fp = open(filename)  # 读取文件
    for i in range(32):
        lineStr = fp.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])  # 将数字取出放入数组中
    fp.close()  # 关闭文件
    return returnVect


def handwritingClassTest(filedir, testdir):
    """
    Description: 验证分类函数的错误率
    Parameters: filedir-文件夹路径 str
    Returns: None
    """
    # 获取所有的文件名
    if not os.path.isdir(filedir):
        print(f'{filedir} 是一个无效的文件夹')
        return

    if not os.path.isdir(testdir):
        print(f'{filedir} 是一个无效的文件夹')
        return

    hwLabels = []
    filedir = os.path.realpath(filedir)
    listFilename = os.listdir(filedir)
    m = len(listFilename)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filename = listFilename[i]
        classNum = int(filename.split('_')[0])
        filename = os.path.join(filedir, filename)
        hwLabels.append(classNum)
        trainingMat[i, :] = img2vector(filename)

    testdir = os.path.realpath(testdir)
    listTestfile = os.listdir(testdir)
    nT = len(listTestfile)
    errorCount = 0.0
    for i in range(nT):
        filename = listTestfile[i]
        classNum = int(filename.split('_')[0])
        filename = os.path.join(testdir, filename)
        vectorUnderTest = img2vector(filename)
        classifierRes = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f'the classifier came back with: {classifierRes}, the real answer is {classNum}')
        if classifierRes != classNum:
            errorCount += 1
    print(f'the total error is: {errorCount}')
    print(f'the total error rate is: {errorCount / float(nT)}')


def plotMat(filename):
    mat, labels = file2matrix(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if False:
        ax.scatter(mat[:, 1], mat[:, 2])
    # 使用标签，标识为不同的大小和颜色
    ax.scatter(mat[:, 0], mat[:, 1], 15.0 * np.array(labels), 15.0 * np.array(labels))
    if False:
        color = []
        for i in labels:
            if i == 1:
                c = 'r'
            elif i == 2:
                c = 'y'
            else:
                c = 'g'
            color.append(c)
        ax.scatter(mat[:, 1], mat[:, 2], 15.0 * np.array(labels), np.array(color))
    plt.show()


def main():
    filename = './dat/datingTestSet2.txt'
    if False:
        datSet, labels = createDatSet()
        inX = [0, 0]
        k = 3
        classify0(inX, datSet, labels, k)
    elif False:
        plotMat(filename)
    elif False:
        datingClassTest(filename)
    elif False:
        while True:
            if classifyPerson(filename):
                break
    else:
        filedir = './dat/trainingDigits'
        testdir = './dat/testDigits'
        handwritingClassTest(filedir, testdir)



if __name__ == '__main__':
    main()
