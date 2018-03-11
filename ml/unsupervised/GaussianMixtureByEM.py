# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import random
def getData():
    n = 300
    # 我们制造两个分三元高斯分类的cluster
    f1_cluster1 = np.round(np.random.normal(1.71, 2.5, n), 5)
    f2_cluster1 = np.round(np.random.normal(8.71, 1.5, n), 5)
    f3_cluster1 = np.round(np.random.normal(9.11, 5.5, n), 5)
    f1_cluster2 = np.round(np.random.normal(12.45, 10.5, n), 5)
    f2_cluster2 = np.round(np.random.normal(1.67, 2.3, n), 5)
    f3_cluster2 = np.round(np.random.normal(4.71, 1.5, n), 5)
    data1 = np.array([f1_cluster1,f2_cluster1,f3_cluster1])
    data2 = np.array([f1_cluster2,f2_cluster2,f3_cluster2])
    data = np.append(data1,data2,axis=1)
    return data


def initParam(data,k):
    '''初始化参数: 潜在变量z的概率分布,k个高斯分布的均值和协方差矩阵'''
    m = data.shape[1]
    n = data.shape[0]
    from sklearn.cluster import KMeans
    # 使用kmeans输出的中心作为数据初始化
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data.T)
    from sklearn.metrics import silhouette_score
    score = silhouette_score(data.T,kmeans.labels_)
    print "kmeans score: %s"%score
    print kmeans.cluster_centers_
    kmeansCentor = kmeans.cluster_centers_.T
    labels = kmeans.labels_
    means = [kmeansCentor[:,i].reshape(n,1) for i in range(0,k)]

    # 使用kmeans分类后的label,得到的两个高斯分布的方差
    vars = []
    dataclusters = []
    for i in range (0,k):
        clusterIndex = []
        for inx,val in enumerate(labels):
            if val==i: # 分类=当前分类
                clusterIndex.append(inx)
        dataclusters.append(clusterIndex)
    for i in range (0,k):
        lenI = len(dataclusters[i])
        dataI = data[:,dataclusters[i]].reshape(n,lenI)
        delMeans = (dataI-means[i])
        varI = (1.0/len(dataI)) * np.dot(delMeans,delMeans.T)
        vars.append(varI)

    pzs = [1.0/k for i in range (1,k+1)]
    return pzs,means,vars

def multipleGaussianProbility(_mean,_var,data):
    '''计算多变量高斯分布的概率'''
    n = data.shape[0]
    det = abs(np.linalg.det(_var))
    coefficient1 = 1.0/(math.sqrt(pow(2*math.pi,n))*math.sqrt(det))
    f1 = data - _mean
    _varInverse = np.asarray(np.asmatrix(_var).I)
    f2 = np.dot(np.dot(f1.T,_varInverse),f1)  # numpy二维矩阵没有求逆运算
    #print f2
    coefficient2 = math.exp((-1.0/2)*f2)
    probility = coefficient1 * coefficient2
    return probility

def e_step(data,m,k,means,vars,pzs):
    '''对样本点进行软估计,估计出样本属于每个高斯分布下,得到z值得后验概率, 即P(Z|X)'''
    posteriorMatrix = np.zeros((k,1))
    n = data.shape[0]
    for i in range(0,m):
        item = data[:,i].reshape(n,1)
        posteriors = []
        pxOnzs = []   # 计算k个p(x|z=k)
        # 如下循环, 计算每个样本点, k个高斯分布下的概率
        for j in range(0,k):
            _mean = means[j]
            _var = vars[j]
            gaussin = multipleGaussianProbility(_mean,_var,item)
            pxOnzs.append(gaussin)
        # 计算P(X)在z条件下的全概率公式
        totalProbilityOfZ = sum([pzs[i]*pxOnzs[i] for i in range(0,k)])
        # 计算每个样本对z的后验概率
        for l in range(0,k):
            posteriors.append([pxOnzs[l]*pzs[l]/totalProbilityOfZ])
        posterior = np.array(posteriors)
        posteriorMatrix = np.append(posteriorMatrix,posterior,axis=1)
    posteriorMatrix = posteriorMatrix[:,1:]
    return posteriorMatrix

def m_step(k,posteriorMatrix,means,data):
    '''更新E_step中的参数, p(z),means,vars'''
    newMeans = []
    newVars = []
    newPzs = []
    m = data.shape[1]
    n = data.shape[0]

    for i in range(0,k):
        ithPosteriors = posteriorMatrix[i,:].reshape(1,m)  # 所有样本的p(z=i)
        ithsum = np.sum(ithPosteriors,axis=1,keepdims=True)
        ithMean = np.sum(np.multiply(data,ithPosteriors),axis=1,keepdims=True)/ithsum
        newMeans.append(ithMean)

        newVar = np.zeros((n,n))
        for j in range(0,m):
            eliminateMean = data[:,j].reshape(n,1) - newMeans[i]
            f1 = np.dot(eliminateMean,eliminateMean.T)*ithPosteriors[:,j][0]
            newVar = newVar + f1
        newVar = newVar*(1.0/ithsum)
        newVars.append(newVar)

        newPz = (1.0/m) * np.sum(ithPosteriors,axis=1,keepdims=True)
        newPzs.append(newPz[0][0])

    return newMeans,newVars,newPzs

def cluster(data,k,means,vars):
    cluster = []
    n = data.shape[0]
    m = data.shape[1]
    for j in range(0,m):
        item = data[:,j].reshape(n,1)
        itemProbs = []
        for i in range(0,k):
            _mean = means[i]
            _vars = vars[i]
            itemProbs.append(multipleGaussianProbility(_mean,_vars,item))
        label = itemProbs.index(max(itemProbs))
        cluster.append(label)
    return cluster


if __name__ == "__main__":
    logData = getData()
    m = logData.shape[1]
    k = 2
    pz,means,vars = initParam(logData,k)

    for i in range(0,30):
        posteriorMatrix = e_step(logData,m,k,means,vars,pz)
        #print posteriorMatrix
        means,vars,pz = m_step(k,posteriorMatrix,means,logData)

    labels = cluster(logData,k,means,vars)
    # 计算选择的类别的平均轮廓系数（mean silhouette coefficient）
    from sklearn.metrics import silhouette_score
    score = silhouette_score(logData.T,labels)
    print "gaussian score: %s"%score
    # 我们发现高斯分布得到的中心店, 更靠近原始生成的两个中心点
    print means

