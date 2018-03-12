# -*- coding: utf-8 -*-
import numpy as np
import math


# 改造最初的逻辑回归, 修改其损失函数为每个样本的加权损失函数
# 注意, 原先的sigmoid函数, 输出为0或1, 而Adaboost需要输出为1和-1
# 因此我们改用双曲正切tanh来进行估值
# TODO tanh作为激活函数的实现, 存在bug, 查看http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf解决方案
def logisticRegression(c, X, y, rate):
    '''c: 每个样本在该分部下的权值矩阵 [c(1),c(2)...c(m)], 用于计算所有样本的损失函数的加权和
       X: 用于训练逻辑回归的输入
       y: 样本真实label
       rate: 学习率'''
    n = X.shape[0]  # feature的数量
    m = X.shape[1]  # 样本数量
    W = np.random.rand(n + 1, 1) * 0.01  # 乘以0.01系数, 是因为我们想让线性方程的值很小, 下降速度快
    # 输入矩阵在第1行增加一个全1行, 即每个向量在列上增加1个0放在第一个
    X0 = [1 for i in range(0, m)]
    X = np.insert(X, 0, values=X0, axis=0)
    assert c.shape == (1, m)

    # 训练逻辑回归
    for i in range(1000):
        # 1. 正向传播
        Z = np.dot(W.T, X)
        A = (2.0 / (1 + np.exp(-4.0/3 * Z)) - 1)*1.7159  # 计算tanh的值, tanh(x)=2sigmoid(2x)-1
        # 2. 反向传播
        # dZ = A - y  # (1,m)
        # dW = np.dot(X, np.multiply(c.T, dZ.T))
        dZ = (np.multiply(2 * A, y) + 2)*1.7159*2/3  # 使用tanh后的偏导
        dW = np.dot(X, np.multiply(c.T, dZ.T)) # Adaboost对每个样本的权值在损失函数偏导数这里体现
        # 3. 梯度下降后更新值
        W -= rate * dW
    return W


# 逻辑回归对样本集的分类结果
def getClassify(X, W):
    '''X: 待预测的样本集合'''
    m = X.shape[1]
    X0 = [1 for i in range(0, m)]
    X = np.insert(X, 0, values=X0, axis=0)  # 输入矩阵在第1行增加一个全1行

    newZ = np.dot(W.T, X)
    newA = 2.0 / (1 + np.exp(-2 * newZ)) - 1  # 学习后的预测值
    return newA


# 计算加权训练误差
def trainError(c, W, X, y):
    '''W: 逻辑回归训练出的参数
       X: 输入矩阵
       y: 真实label'''
    predicts = getClassify(X, W)
    predicts[predicts >= 0] = 1
    predicts[predicts < 0] = -1
    # 分类错误,无非是y=-1,而预测1,或y=1,预测-1, 所有其差值只能为正负2
    # 取绝对值后除以2就能让预测失败的样本在该值取1, 方便后面的加权矩阵乘
    error = 0.5 * np.abs(y - predicts)
    weightError = np.dot(c, error.T)
    return weightError, predicts


# Adaboost算法:
def adaBoostTrain(X, y, k):
    '''使用k个弱分类器'''
    m = X.shape[1]
    # result = np.zeros((1, m))  # 最终Adaboost返回的结果
    functionFactors = []
    Ws = []

    # 1.初始化样本权值
    cList = [1.0 / m for i in range(0, m)]
    c = np.array([cList])

    for i in range(0, k):
        # 2.训练弱分类器
        W = logisticRegression(c, X, y, rate=0.2)
        Ws.append(W)
        # 3.计算弱分类器训练误差及改弱分类器的线性权值
        error, predicts = trainError(c, W, X, y)
        # 由错误率得到该弱分类器在adaboost最终结果中所占权值
        functionFactor = 1.0 / 2 * math.log((1 - error) / error)
        functionFactors.append(functionFactor)
        # 4.更新样本分布的权值
        matrix1 = np.multiply(predicts, y) * functionFactor * (-1)
        matrix2 = np.multiply(c, np.exp(matrix1))
        sum = np.sum(matrix2, axis=1, keepdims=True)[0][0]
        c = (1.0 / sum) * matrix2
        # 5.计算改弱分类器的加权结果, sum入总结果中
        # result = result + np.multiply(functionFactor, predicts)

    return functionFactors, Ws


def adaBoostClassify(X_test, functionFactors, Ws):
    k = len(functionFactors)
    m = X_test.shape[1]
    result = np.zeros((1, m))  # 最终Adaboost返回的结果
    for i in range(0, k):
        predicts = getClassify(X_test, Ws[i])
        result = result + np.multiply(functionFactors[i], predicts)
    result[result < 0] = -1
    result[result >= 0] = 1
    return result


if __name__ == "__main__":
    from preproccess.HandleDatasets import getDataset
    from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

    X_train, X_val, X_test, y_train, y_val, y_test = getDataset()
    X_train, X_val, X_test, y_train, y_val, y_test = X_train.as_matrix().T, X_val.as_matrix().T, X_test.as_matrix().T, \
                                                     y_train.as_matrix().T, y_val.as_matrix().T, y_test.as_matrix().T
    k = 2
    y_train = y_train * 1.0
    y_train[y_train == 0] = -1
    functionFactors, Ws = adaBoostTrain(X_train, y_train, k)
    print "functionFactors: %s" % functionFactors
    predicts = adaBoostClassify(X_train, functionFactors, Ws)  # 使用3个弱分类器
    y_train, newA = y_train.flatten(), predicts.flatten()  # 把二维转成一维
    print "Adaboost by logisticRegression : %s" % k
    print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_train, newA))
    print "F-score on validation data: {:.4f}".format(fbeta_score(y_train, newA, beta=0.5))
    print confusion_matrix(y_train, newA)

    # TODO 以上使用tanh作为激活函数的实现, 存在bug, 学习效果不佳, 需进一步查看解决方法
