from math import log
import operator as op
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet,labels

"""
熵
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)

    return shannonEnt

# dataSet,lables = createDataSet()
# print(calcShannonEnt(dataSet))

def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet

# dataSet,lables = createDataSet()
# print(splitDataSet(dataSet,1,1))

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

# dataSet,lables = createDataSet()
# # # print(chooseBestFeatureToSplit(dataSet))

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(),
                              key=op.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels,featLabels):
    classList = [example[-1] for example in dataSet]
    #只剩一个类型了 返回就行啦
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #所有特征都遍历完了，还是有1个以上的类型
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)

    return myTree

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if __name__ == '__main__':
            if type(secondDict[key]).__name__=='dict':
                numLeafs ++ getNumLeafs(secondDict[key])
            else:
                numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth

    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 绘制结点createPlot.ax1创建绘图区
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    # 计算标注位置（箭头起始位置的中点处）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    # 设置结点格式boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        # 如果是叶结点，绘制叶结点，并标注有向边属性值
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD


def createPlot(inTree):
    # 创建fig
    fig = plt.figure(1, facecolor="white")
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()
"""
{
    "有自己的房子": {
        "0": {
            "有工作": {
                "0": "no", 
                "1": "yes"
            }
        }, 
        "1": "yes"
    }
}
"""
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]

    return classLabel




dataSet,labels = createDataSet()
featLabels = []
myTree = createTree(dataSet,labels,featLabels)
# # createPlot(myTree)
testDataSet = [0, 1, 1, 1]
print(classify(myTree,featLabels,testDataSet))




