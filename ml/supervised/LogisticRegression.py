# -*- coding: utf-8 -*-
# 本文件实现逻辑回归
import numpy as np
from preproccess.HandleDatasets import getDataset
# 逻辑回归矩阵实现

X_train, X_val, X_test, y_train, y_val, y_test = getDataset()
X_train, X_val, X_test, y_train, y_val, y_test = X_train.as_matrix().T,X_val.as_matrix().T, X_test.as_matrix().T, \
                                                 y_train.as_matrix().T, y_val.as_matrix().T, y_test.as_matrix().T
print(X_train.shape)

# 1. 变量声明
m = X_train.shape[1]  # 样本数量
n = X_train.shape[0]  # 特征数量
rate = 0.2

W = np.zeros((n,1),dtype='float64')  # 参数矩阵
B = np.zeros((1,1),dtype='float64')  # B = np.array([[ 1.0,  1.0]])#

Z = np.zeros((1,m),dtype='float64')
A = np.zeros((1,m),dtype='float64')
dZ = np.zeros((1,m),dtype='float64')
db = np.zeros((1,1),dtype='float64')
dW = np.zeros((n,1),dtype='float64')

# 2. Logistic正反传播
for i in range(1000):
    # 1. 正向传播
    Z = np.dot(W.T,X_train)+B
    A = 1.0/(1+np.exp(-Z))    # 计算sigmoid的值
    # 2. 反向传播
    dZ = A-y_train   #(1,m)
    db = np.sum(dZ)*(1.0/m)
    dW = np.dot(X_train,dZ.T)*(1.0/m)
    # 3. 梯度下降后更新值
    W -= rate*dW
    B -= rate*db

newZ = np.dot(W.T,X_test)+B
newA = 1.0/(1+np.exp(-newZ))  # 学习后的预测值
newA[newA<=0.5]=0
newA[newA>0.5]=1

# 3. 判断算法性能
from sklearn.metrics import fbeta_score,accuracy_score
y_test,newA = y_test.flatten(), newA.flatten() # 把二维转成一维
print "y_test shape: %s" % y_test.shape #(9045,)
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_test, newA))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_test, newA, beta = 0.5))