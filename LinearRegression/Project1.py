# 
"""
@author: xxw
@time: 2020/3/18 15:51
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr,yArr


def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    ws = (xTx.I) * (xMat.T) * yMat
    return ws

def plotDataSet():
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()

    xCopy.sort(0)
    yHat = xCopy * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')

    ax.scatter(xMat[:,1].flatten().A[0],yMat.flatten().A[0],s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotDataSet()