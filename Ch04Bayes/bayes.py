#!/usr/bin/python3
# -*-coding:utf-8-*-

# **********************************************************
# * Author        : SnailCL
# * Create time   : 2020-08-27 08:28:02
# * Last modified : 2020-08-27 08:28:02
# * Filename      : bayes.py
# * Description   :
# **********************************************************

import numpy as np
import re
import random


def loadDateSet():
    """
    """
    postingList = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    """
    """
    vocabSet = set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f'the word: {word} is not in my vocabulary!')
    return returnVec


def bagOfWords2Vec(vocabList, inputSet):
    """
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print(f'the word: {word} is not in my vocabulary!')
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / numTrainDocs
    p0Num, p1Num = np.zeros(numWords), np.zeros(numWords)
    p0Denom, p1Denom = 0.0, 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def trainNB1(trainMatrix, trainCategory):
    """
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / numTrainDocs
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    """
    """
    p1 = sum(vec2Classify * p1Vect) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    postingList, classVec = loadDateSet()
    myVocabList = createVocabList(postingList)
    trainMat = []
    for posting in postingList:
        # trainMat.append(setOfWords2Vec(myVocabList, posting))
        trainMat.append(bagOfWords2Vec(myVocabList, posting))
    p0V, p1V, pAb = trainNB1(np.array(trainMat), np.array(classVec))

    testEntry = ['love', 'my', 'dalmation']
    # thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
    print(f'{testEntry} classify as: {classifyNB(thisDoc, p0V, p1V, pAb)}')

    testEntry = ['stupid', 'garbage']
    # thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    thisDoc = np.array(bagOfWords2Vec(myVocabList, testEntry))
    print(f'{testEntry} classify as: {classifyNB(thisDoc, p0V, p1V, pAb)}')


def textParse(bigString):
    """
    """
    listOfTakens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTakens if len(tok) > 2]


def spamTest():
    docList, classList, fullText = [], [], []
    for i in range(1, 26):
        filename = f'./dat/email/spam/{i}.txt'
        # print(filename)
        with open(filename, 'r', encoding='utf8') as fp:
            wordList = textParse(fp.read())

        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)

        filename = f'./dat/email/ham/{i}.txt'
        # print(filename)
        with open(filename, 'r', encoding='utf8') as fp:
            wordList = textParse(fp.read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet, testingSet = list(range(50)), []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testingSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB1(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testingSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print(f'the error rate is: {errorCount / len(testingSet)}')


def main():
    if False:
        postingList, classVec = loadDateSet()
        myVocabList = createVocabList(postingList)
        for posting in postingList:
            print('posting vect:', setOfWords2Vec(myVocabList, posting))
    elif False:
        trainMat = []
        postingList, classVec = loadDateSet()
        myVocabList = createVocabList(postingList)
        for posting in postingList:
            trainMat.append(setOfWords2Vec(myVocabList, posting))
        p0V, p1V, pAb = trainNB0(trainMat, classVec)
        print(p0V, p1V, pAb)
        p0V, p1V, pAb = trainNB1(trainMat, classVec)
        print(p0V, p1V, pAb)
    elif False:
        testingNB()
    else:
        spamTest()


if __name__ == '__main__':
    main()
