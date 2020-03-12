# 
"""
@author: xxw
@time: 2020/3/11 16:28
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt
"""
求 f(x) = -x^2 + 4x 的最大值
"""
def Gradient_test():
    def f_prime(x_old):
        return -2*(x_old) +4

    x_old = -1
    #
    x_new = 0
    #步长
    alpha = 0.01
    presision = 0.0000001

    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)

    print("x_new:"+str(x_new))
# print("sdfsd")
# Gradient_test()


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    # 100*3
    dataMatrix = np.mat(dataMatIn)
    # 100 * 1
    labelMat = np.mat(classLabels).transpose()
    # 100,3
    m,n = np.shape(dataMatrix)
    alphha = 0.001
    maxCycles = 500
    # 3 * 1   all number is 1
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # 100 * 1
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alphha * dataMatrix.transpose() * error

    return weights


def plotBestFit(wei):
    weights = wei.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)

    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGraAscent(dataMatrix,classLabels):
    # 100,3
    m,n = np.shape(dataMatrix)
    alphha = 0.001
    # 数组 3
    weights = np.ones(3)
    for k in range(m):
        # 100 * 1
        h = sigmoid(dataMatrix[k] * weights)
        error = (classLabels[k] - h)
        weights = weights + alphha * dataMatrix[k].transpose() * error

    return weights




dataMat, labelMat = loadDataSet()
# weights2 = gradAscent(dataMat, labelMat)

weights2 = stocGraAscent(np.array(dataMat),labelMat)
print(weights2)

plotBestFit(weights2)








