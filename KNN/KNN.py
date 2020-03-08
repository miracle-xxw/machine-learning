from array import array
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX,dataSet,labels,k):
    #取第一维度的值1
    dataSetSize = dataSet.shape[0]
    #tile函数[inX]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    #axis=0按列相加，axis=1按行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

# def createDataSet():
# group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
# labels = ['A','A','B','B']
# print(classify0([0,0],group,labels,3))

#解析文件
def fine2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    #返回矩阵 numberOfLines行，3列
    returnMat = np.zeros((numberOfLines,3))

    classLabelVector = []
    index = 0

    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector

#print(fine2matrix('datingTestSet2.txt'))
#datingDataMat, datingLabels = fine2matrix('datingTestSet2.txt')
#print(datingDataMat)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0],datingDataMat[:,1])
# plt.show()

##归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))

    return normDataSet,ranges,minVals

#normMat,ranges,minVals = autoNorm(datingDataMat)
#print(normMat)

def datingClassTest():
    filename = "datingTestSet2.txt"
    datingDataMat, datingLabels = fine2matrix(filename)
    hoRatio = 0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:]
                                    ,datingLabels[numTestVecs:m],4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("错误率:%f%%" % (errorCount/float(numTestVecs)))

datingClassTest()



