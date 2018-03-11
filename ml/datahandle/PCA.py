# -*- coding: utf-8 -*-
import numpy as np
# PCA
# X: 观测矩阵
# newX: 待转换的新样本
X = np.array([[1,2,3],[2,3,5],[5,6,7],[2,3,2]])  # 4个变量,3个特征
newX = np.array([[2,1,3]])

# 首先使用sklearn做PCA转换, 与下面进行对比
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print pca.transform(newX)
radio = pca.explained_variance_ratio_
print(radio)

print ("================")
#  自己实现
# sklearn api均已行为一个样本, 下面计算使用列作为一个样本
X = X.T
##(1)计算协方差矩阵 :
 #   先把观测矩阵化为平均偏差形式: B = X - Xbar
 #   在计算平均偏差下的协方差矩阵: varX= B * B.T *(1/n).
Xbar = (1.0/4)*np.sum(X,axis=1,keepdims=True)
print Xbar
B = X-Xbar # 平均偏差形式
varX = np.dot(B,B.T)*(1.0/4)  # 协方差矩阵 = (1/n)* B*B.T

##(2)计算协方差矩阵的特征向量,后从大到小排列,得到解
print "协方差矩阵的特征值:"
eigVal = np.linalg.eig(varX)[0]  # 特征值 , 已按照从大到小排列 [  7.47698648e+00   1.56366740e-16   7.10513523e-01]
# eigVal = eigVal.reshape(3,1)
print(eigVal)
print("第一个主成分占总方差的: %s" % (eigVal[0]/np.sum(eigVal)))  # 对比radio[0]
p = np.linalg.eig(varX)[1]  # 特征向量矩阵
index = np.argsort(-eigVal) # 特征值从大到小排序的序号
print "协方差矩阵特征向量: "
p = p[:,index]  # 安札特征值从大到小排序后的特征向量重组矩阵
print p
# 如下为p的每一列,即协方差矩阵对应的特征向量, 每个特征向量都是原观测矩阵的Feature的一个线性组合
# sklearn中n_components属性为取前n特征向量, 对feature线性组合后的结果
# [[ -5.29168529e-01  -4.69020967e-01  -7.07106781e-01]
#  [ -5.29168529e-01  -4.69020967e-01   7.07106781e-01]
# [ -6.63295813e-01   7.48357311e-01  -1.26760092e-16]]

print ("my methods :")   # 取得的值与sklearn一致
print np.dot(p.T,(Xbar-newX.T))
# [[ 2.41662535]
#  [-1.41421356]
#  [ 0.47161626]]