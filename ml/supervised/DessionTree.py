# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import copy

# 给dataframe的所有字段进行数值编码,将不同类型转换为数字编码
def mark_Label(markList, data):
    from sklearn import preprocessing
    encoder = preprocessing.LabelEncoder()
    encoder.fit_transform(data)
    markList.append(encoder.classes_)


def getData():
    data = pd.read_csv("../../datasets/titanic.csv")
    survived = data['Survived'].to_frame("survived")  # pandas取到的列为Series, 将其转换成Dataframe,在加上列名
    features = data.drop(['PassengerId', 'Survived', 'Ticket', 'Age', 'Name', 'Cabin', 'Fare'], axis=1)
    # print data.SibSp.value_counts()  查看每个字段有那几个取值,每个取值有多少样本
    # print features.head()

    # 字段标签编码
    from sklearn import preprocessing
    sexEncoder = preprocessing.LabelEncoder()
    features['Sex'] = sexEncoder.fit_transform(features['Sex'])
    embarkEncoder = preprocessing.LabelEncoder()
    features['Embarked'] = embarkEncoder.fit_transform(features['Embarked'])
    outEncoder = preprocessing.LabelEncoder()  # sklearn编码后,将df转换成ndarray
    survived = pd.DataFrame(outEncoder.fit_transform(survived), columns=["Survived"])

    # print "After label encoder : "
    print features.head()
    # print "Survived: "
    # print survived.head()

    # 切分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, survived, test_size=0.2, random_state=0,
                                                        stratify=survived)  # 切片以后仍是dataframe
    return X_train, X_test, y_train, y_test


def infEntropy(labels):
    '''计算信息熵'''
    # [np.newaxis,:]用于把一维ndarray转换成二维
    labelArray = labels.Survived.value_counts().values[np.newaxis,:]
    # print labelArray
    sum = np.sum(labelArray,axis=1,keepdims=True)[0][0]
    probility = labelArray*(1.0/sum)
    # print probility
    logProbility = np.log2(probility).T*(-1)
    entropy = np.dot(probility,logProbility)[0][0]
    # print entropy
    return entropy

def infGain(parentEntropy, X, y, param):
    '''消息增益 = 父项墒 - sum{分类百分比*分类后的墒}'''
    groupDict = X.groupby(param).indices
    sumCount = X.shape[0]
    f1 = parentEntropy
    f2 = []
    for k,v in groupDict.items(): # k为group的feature取值,v为当前值包括的index(ndarray形)
        indices = v
        sliceLabel = y.iloc[indices]
        varietyCount = indices.shape[0]
        proportion = 1.0*varietyCount/sumCount # 当前属性的取值所占百分比
        entropy = infEntropy(sliceLabel)
        f2.append(proportion*entropy)
    gain = f1 - sum(f2)
    # print gain
    return gain

def decessionTree(parentEntropy,featureNames,X,y,level):
    '''featureNames: 属性集
        X: 输入集合
        y: 输出集合'''
    featureNames = copy.deepcopy(featureNames)
    kinds = y.Survived.value_counts().shape[0] # 对Seriers进行value_counts后,仍是Seriers: 单列,其shape为(行数,)
    if kinds==1: # 若当前样本,已经全部属于同一类别,则返回这些样本的label(决策完毕)
        print "当前样本,已经全部属于同一类别"
        return y.Survived.values[0] #按照index取值, 转换为ndarray用角标取值
    if len(featureNames)==0: # 如果已无属性可用,则返回当前样本中,占比最多的分类
        print "已无属性可用"
        return y.Survived.value_counts().index[0]

    ## 找到消息增益最大的那个feature
    featureGains = []
    for featureName in featureNames:
        featureGains.append(infGain(parentEntropy,X,y,featureName))
    index = featureGains.index(max(featureGains))
    maxGainFeature = featureNames[index]
    print "current max gain feature : %s" % maxGainFeature

    ## 按照最大增益feature划分样本, 寻找下层最大增益的属性
    groupDict = X.groupby(maxGainFeature).indices
    for k,v in groupDict.items():
        print "current branch : %s = %s, level = %s" % (maxGainFeature,k,level)
        indices = v  # 在最大增益属性上取值相同的样本index集合
        sliceLabel = y.iloc[indices]
        sliceSample = X.iloc[indices]
        currentEntropy = infEntropy(sliceLabel)
        curentFeature = copy.deepcopy(featureNames)
        curentFeature.remove(maxGainFeature) # 该属性已使用完毕
        decessionTree(currentEntropy,curentFeature,sliceSample,sliceLabel,level+1)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = getData()
    # 初始化父项墒, 树层级为0
    featureNames = X_train.columns.values.tolist()
    initialEntropy = infEntropy(y_train)
    decessionTree(initialEntropy,featureNames,X_train,y_train,0)
