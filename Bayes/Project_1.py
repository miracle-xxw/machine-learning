# 
"""
@author: xxw
@time: 2020/3/9 16:41
@desc: 
"""
import numpy as np
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in ")

    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p0Vect = p0Num/p0Denom
    p1Vect = p0Num/p0Denom
#    p1Vect = np.log(p1Num/p1Denom)
#    p0Vect = np.log(p0Num/p0Denom)


    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bagOfWords2Vec(myVocabList, postinDoc))
    p0Vect, p1Vect, pAbusive = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(bagOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc, p0Vect, p1Vect, pAbusive):
        # 执行分类并打印结果
        print(testEntry, '属于侮辱类')
    else:
        # 执行分类并打印结果
        print(testEntry, '属于非侮辱类')


# postingList, classVec = loadDataSet()
# myVocabList = createVocabList(postingList)
# print(myVocabList)
# print(len(myVocabList))
# #print(setOfWords2Vec(myVocabList,postingList[2]))
# trainMat=[]
# for postinDoc in postingList:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
# # print(trainMat)
# # print(len(trainMat))
# # print(len(trainMat[0]))
# # print(len(trainMat[1]))
# # print(len(trainMat[2]))
# # print(len(trainMat[3]))
# p0Vect,p1Vect,pAbusive = trainNB0(trainMat,classVec)
# print(p0Vect)
# print(p1Vect)
# print(pAbusive)
#
testingNB()








