# 
"""
@author: xxw
@time: 2020/3/13 15:03
@desc: 
"""
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat,labelMat


def selectJrand(i,m):
    j = i
    while(j == i ):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return  aj

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    # 100 * 2
    dataMatrix = np.mat(dataMatIn)
    # 100 * 1
    labelMat = np.mat(classLabels).transpose()
    b = 0
    # 100,2
    m,n = np.shape(dataMatrix)
    # 100 * 1
    alphas = np.mat(np.zeros((m,1)))
    iter_num = 0
    while(iter_num < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
