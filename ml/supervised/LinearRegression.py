# -*- coding: utf-8 -*-
# 本文实现线性回归预测波士顿房价

import numpy as np
import pandas as pd
def getData():
    data = pd.read_csv("../datasets/boston.csv")
    m = data.shape[0] # 输入个数
    price_raw = data['price'].as_matrix().reshape(m,1)
    features_raw = data.drop('price',axis=1).as_matrix()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler() # 此处进行数据的归一化,才能适应后面的线性回归, 不至于多次循环后出现Nan问题
    price_raw = scaler.fit_transform(price_raw)
    features_raw = scaler.fit_transform(features_raw)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features_raw, price_raw, test_size = 0.2, random_state = 0)
    return X_train.T, X_test.T, y_train.T, y_test.T

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = getData()
    X_train = np.insert(X_train,0,values=1,axis=0) # 第一行增加x0=1
    #y_train = y_train.T
    # 1. 变量声明
    m = X_train.shape[1]  # 样本数量
    n = X_train.shape[0]  # 特征数量
    rate = 0.2

    W = np.zeros((n,1),dtype="float64")
    # 梯度下降
    for i in range(10000):
        A = np.dot(X_train.T,W).T
        # print A-y_train
        # print np.sum(np.multiply((A-y_train),X_train),axis=1,keepdims=True)
        dW = (1.0/m)*np.sum(np.multiply((A-y_train),X_train),axis=1,keepdims=True) # 按行相加
        W -= rate*dW
    print "系数矩阵为: "
    print W.T
    print "回归的输入feature:"
    print X_train[:,0]
    print "回归的预测输出: "
    print np.dot(X_train[:,0],W)
    print "实际输出值: "
    print(y_train[:,0])