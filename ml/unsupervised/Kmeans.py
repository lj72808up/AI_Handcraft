# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
def getData():
    data = pd.read_csv("../datasets/cluster.csv")
    data = data.drop("cluster",axis=1)
    # 如果数据不是正态分布的，尤其是数据的平均数和中位数相差很大的时候（表示数据非常歪斜）。
    # 这时候通常用一个非线性的缩放是很合适的，尤其是对于金融数据。
    # 一种实现这个缩放的方法是使用Box-Cox 变换，这个方法能够计算出能够最佳减小数据倾斜的指数变换方法。
    # 一个比较简单的并且在大多数情况下都适用的方法是使用自然对数。
    logData = np.nan_to_num(np.log(data))  # 把np.inf与np.nan设置成很小的数和很大的数
    return logData.T

# 初始化中心点
def initCentorIds(m,k,data):
    centroIds = []
    for i in range(0,k):
        # random.seed(i)
        # index = random.randint(0, m)
        # centroIds.append(data[:,index])
        centroIds.append(data[:,i])
    return centroIds

# 计算个样本属于哪个cluster
def ownClusterId(data,centroIds):
    columns = data.shape[1]
    ownCentroIds = []
    for i in range(0,columns): # 遍历每一列
        item = data[:,i]
        dists = [np.linalg.norm(item-centroId) for centroId in centroIds]  # 计算当前项与每个centroid的距离
        ownId = dists.index(min(dists))  # 找到最小距离所属的centroId
        ownCentroIds.append(ownId)
    return ownCentroIds


if __name__ == "__main__":
    data = getData()
    k = 3  # 分3类
    m = data.shape[1] # 样本数量
    # 1. 初始化中心点
    centroIds = initCentorIds(m,k,data)
    print centroIds
    for i in range(100):
        # 2. 计算每个样本所属中心点
        ownIds = ownClusterId(data,centroIds)
        # 3. 把所属同个centro的点,重新计算其中心位置
        clusterIndexs = []
        for i in range(0,k):
            sameClusterIndex = [j for j,val in enumerate(ownIds) if val==i]
            clusterIndexs.append(sameClusterIndex)
        # 此时, clusterIndex为二维数组,[[cluster0的所有index],[cluster1的所有index],[cluster2的所有index]]
        newCentroIds = np.zeros((data.shape[0],1))
        for clusterindex in clusterIndexs:
            clusterNode = data[:,clusterindex]
            num = clusterNode.shape[1]
            newCentro = (1.0/num) * np.sum(clusterNode,axis=1,keepdims=True)
            newCentroIds = np.append(newCentroIds,newCentro,axis=1)
        newentroIds = newCentroIds[:,1:]
        # 4. 整理格式,让newentroIds符合centroIds的形式继续循环
        b = []
        for i in range(k):
            b.append(newentroIds[:,i])
        centroIds = b

    print centroIds
    # 计算选择的类别的平均轮廓系数（mean silhouette coefficient）
    from sklearn.metrics import silhouette_score
    labels = ownClusterId(data,centroIds)
    score = silhouette_score(data.T,labels)
    print "score: %s"%score