#!/usr/bin/python3
# -*-coding:utf-8-*-

# **********************************************************
# * Author        : SnailCL
# * Create time   : 2020-08-26 08:26:38
# * Last modified : 2020-08-26 20:11:25
# * Filename      : trees.py
# * Description   :
# **********************************************************

import math
import operator
import pickle
from datetime import datetime

from treePlotter import createPlot


def createDatSet():
    """
    """
    datSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return datSet, labels


def calcShannonEnt(datSet):
    """
    """
    numEntries = len(datSet)
    labelCount = {}
    for featVec in datSet:
        currentLabel = featVec[-1]
        labelCount[currentLabel] = labelCount.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / numEntries  # 选择该分类的概率
        shannonEnt -= prob * math.log2(prob)    # 累加信息期望, 得到熵
    return shannonEnt


def splitDatSet(datSet, axis, value):
    """
    """
    retDatSet = []
    for featVec in datSet:
         if featVec[axis] == value:
             reduceFeatVec = featVec[:axis]
             reduceFeatVec.extend(featVec[axis+1:])
             retDatSet.append(reduceFeatVec)
    return retDatSet


def chooseBestFeatureToSplit(datSet):
    """
    """
    numFeatures = len(datSet[0]) - 1                        # 除去标签外的特征个数
    baseEntropy = calcShannonEnt(datSet)                    # 数据集的熵
    bestInfoGain, bestFeature = 0.0, -1                     # 初始化信息增益，最佳特征index
    for i in range(numFeatures):
        featList = [example[i] for example in datSet]       # 依次获取特征值
        uniqueVal = set(featList)                           # 特征值唯一化
        newEntropy = 0.0
        for value in uniqueVal:
            subDatSet = splitDatSet(datSet, i, value)       # 按特征划分子集
            prob = len(subDatSet) / len(datSet)             # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDatSet)  # 计算条件熵
        infoGain = baseEntropy - newEntropy                 # 本特征下的信息增益
        if infoGain > bestInfoGain:                         # 更新最佳信息增益和最佳特征的索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 返回标签出现最多次数的类别


def createTree(datSet, labels):
    """
    """
    classList = [example[-1] for example in datSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同则停止划分
    if len(datSet[0]) == 1:  # 遍历完所有的特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(datSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    # 获取列表包含的所有属性值
    featValues = [example[bestFeat] for example in datSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDatSet(datSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    """
    """
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)
    print('将决策树序列化')


def grabTree(filename):
    """
    """
    try:
        with open(filename, 'rb') as fr:
            return pickle.load(fr)
    except Exception:
        return


def text2DatSet(filename):
    with open(filename, 'r') as fp:
        lenses = [inst.strip().split('\t') for inst in fp.readlines()]

    lensesLabels = ['age','prescript', 'astigmatic', 'tearRate']
    spt1 = datetime.now()
    lenseTree = grabTree('./dat/lenseTree.dat')
    spt2 = datetime.now()
    print(f'grabTree {(spt2 - spt1).microseconds}毫秒')
    if lenseTree is None:
        lenseTree = createTree(lenses, lensesLabels)
        spt3 = datetime.now()
        print(f'createTree {(spt3 - spt2).microseconds}毫秒')
        storeTree(lenseTree, './dat/lenseTree.dat')
        spt4 = datetime.now()
        print(f'storeTree {(spt4 - spt3).microseconds}毫秒')
    createPlot(lenseTree)


def test_createDatSet():
    dataSet = [
        [2, 1, 1, 0, 'love'],
        [2, 0, 1, 0, 'love'],
        [1, 1, 1, 0, 'love'],
        [2, 1, 1, 1, 'love'],
        [2, 1, 1, 2, 'love'],
        [2, 0, 0, 1, 'like'],
        [2, 0, 1, 1, 'like'],
        [2, 0, 0, 0, 'like'],
        [2, 0, 1, 2, 'like'],
        [2, 1, 0, 0, 'like'],
        [2, 1, 0, 1, 'like'],
        [1, 1, 0, 0, 'like'],
        [1, 1, 1, 1, 'like'],
        [0, 1, 1, 0, 'like'],
        [0, 1, 1, 1, 'like'],
        [2, 1, 0, 2, 'like'],
        [1, 1, 1, 2, 'like'],
        [0, 1, 1, 2, 'like'],
        [1, 0, 1, 2, 'like'],
        [1, 1, 0, 2, 'like'],
        [0, 0, 1, 2, 'like'],
        [1, 0, 1, 0, 'like'],
        [1, 0, 1, 1, 'like'],
        [0, 0, 1, 0, 'like'],
        [0, 0, 1, 1, 'like'],
        [1, 1, 0, 1, 'like'],
        [0, 1, 0, 0, 'dislike'],
        [0, 1, 0, 1, 'dislike'],
        [0, 1, 0, 2, 'dislike'],
        [0, 0, 0, 0, 'dislike'],
        [0, 0, 0, 1, 'dislike'],
        [1, 0, 0, 0, 'dislike'],
        [1, 0, 0, 1, 'dislike'],
        [2, 0, 0, 2, 'dislike'],
        [1, 0, 0, 2, 'dislike'],
        [0, 0, 0, 2, 'dislike'],
    ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels  #返回数据集和分类属性


def test_creatTree():
    datSet, labels = test_createDatSet()
    if False:
        myTree = createTree(datSet, labels)
        storeTree(myTree, './dat/myTree.dat')
    else:
        myTree = grabTree('./dat/myTree.dat')
        for testVec in datSet:
            res = classify(myTree, labels, testVec)
            print(f'classify result is {res}, real value is {testVec[-1]}')


def test_useTree():
    myTree = grabTree('./dat/myTree.dat')
    if 0:
        print(myTree)
        createPlot(myTree)  # 可视化决策树
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    while True:
        try:
            q1 = int(input('年龄:'))
            if 20 < q1 <= 30:
                q1 = 2
            elif 30 < q1 <= 50:
                q1 = 1
            else:
                q1 = 0

            q2 = int(input('是否有工作(0没有/1有):'))
            assert q2 in [0, 1]

            q3 = int(input('是否有自己的房子(0没有/1有):'))
            assert q3 in [0, 1]

            q4 = int(input('信贷情况(0少/1中/2多):'))
            assert q4 in [0, 1, 2]

            testVec = [q1, q2, q3, q4]
            res = classify(myTree, labels, testVec)
            print(f'classify result is {res}')
        except Exception:
            break


def main():
    """
    """
    # text2DatSet('./dat/lenses.txt')
    # test_creatTree()
    test_useTree()


if __name__ == '__main__':
    main()
