一. 机器学习算法实现
----------------
使用pandas, numpy实现几种机器学习算法  
使用sklearn的评分包对算法进行评价  

- [0. PCA](ml/datahandle/PCA.py)  
- [1. 决策树](ml/supervised/DessionTree.py)  
- [2. 线性回归](ml/supervised/LinearRegression.py)  
- [3. 逻辑回归](ml/supervised/LogisticRegression.py)  
- [4. 高斯混合模型](ml/unsupervised/GaussianMixtureByEM.py)  
- [5. Kmeans](ml/unsupervised/Kmeans.py)  
- [6. Adaboost-线性加权lR](ml/supervised/AdaBoost.py) - tanh激活函数待改善  
- [7. gbdt-pipeline,网格搜索](ml/gbdt/GbdtDemo.py)  
- [8. xgboost](ml/gbdt/XgboostDemo.py)  

二. 数据处理
----------------
#### 2.1 Instance
- [matplotlab - 泰坦尼克生还者](preproccess/TitanicPlot.py)
- [数据编码,倾斜处理](preproccess/HandleDatasets.py)

#### 2.2 pandas数据预处理
- [1-pandas数据结构.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/1-pandas%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84.ipynb)
- [2-pandas数据加载与文件格式.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/2-pandas%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E4%B8%8E%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F.ipynb)
- [3-清洗数据与数据准备.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/3-%E6%B8%85%E6%B4%97%E6%95%B0%E6%8D%AE%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.ipynb)
- [4-聚合,合并,重塑.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/4-%E8%81%9A%E5%90%88%2C%E5%90%88%E5%B9%B6%2C%E9%87%8D%E5%A1%91.ipynb)
- [5-matplotlib绘图与可视化.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/5-%E7%BB%98%E5%9B%BE%E4%B8%8E%E5%8F%AF%E8%A7%86%E5%8C%96.ipynb)
- [6-pandas与seaborn高级绘图.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/6-seaborn%E9%AB%98%E7%BA%A7%E7%BB%98%E5%9B%BE.ipynb)
- [7-聚合与分组.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/7-%E8%81%9A%E5%90%88%E4%B8%8E%E5%88%86%E7%BB%84.ipynb)
- [8-pandas时间序列.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/8-pandas%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97.ipynb)
- [9-pandas高级应用.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/79a8d722ec8d65deffc937ae5615f0cbca0219b2/blog/pandas%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86/9-pandas%E9%AB%98%E7%BA%A7%E5%BA%94%E7%94%A8.ipynb)


三. Blog
--------------------------------
- [Jupyter-spark环境](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/Jupyter-spark%E9%85%8D%E7%BD%AE.ipynb)
- [0-数据预处理.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/0-%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.ipynb)
- [2-线性回归与逻辑回归](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/d79fdd7d50ddffcb1b81abbcdceb6974a476a628/blog/2-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92.ipynb)
- [3-生成模型(高斯判别与朴素贝叶斯)](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/d79fdd7d50ddffcb1b81abbcdceb6974a476a628/blog/3-%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%28%E9%AB%98%E6%96%AF%E5%88%A4%E5%88%AB%2C%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%29.ipynb)
- [4-svm](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/4-svm.ipynb)
- [2-CART决策树.ipynb](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/5-决策树.ipynb)

#### 3.1 聚类与检索  
- [1-聚类与检索(KNN,TF-IDF,KD-Tree,K-Means,EM]
    * [1.1 K-NN](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/d5bdd5a4571dc2e0b491ea5b6585241a4dfc106f/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/1-1%20%20KNN%2CTF-IDF.ipynb)
    * [1.2 KD树](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/d5bdd5a4571dc2e0b491ea5b6585241a4dfc106f/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/1-2%20KD%E6%A0%91.ipynb)
    * [1.3 局部敏感哈希](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/d5bdd5a4571dc2e0b491ea5b6585241a4dfc106f/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/1-3%20%E5%B1%80%E9%83%A8%E6%95%8F%E6%84%9F%E5%93%88%E5%B8%8C.ipynb)
    * [1.4 KMeans](https://github.com/lj72808up/ML_Handcraft/blob/d5bdd5a4571dc2e0b491ea5b6585241a4dfc106f/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/1-4%20Kmeans.ipynb)
    * [1.5 搞死混合EM](https://github.com/lj72808up/ML_Handcraft/blob/d5bdd5a4571dc2e0b491ea5b6585241a4dfc106f/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/1-5%20%E6%B7%B7%E5%90%88%E9%AB%98%E6%96%AFEM.ipynb)
- [2-LDA与混合关系模型](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/2-LDA.ipynb)
- [3-Hierarchical Clustering](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/174c79b7b0f989818c8edcd63b45512e02f2c87e/blog/%E8%81%9A%E7%B1%BB%E4%B8%8E%E6%A3%80%E7%B4%A2/3-Hierarchical%20Clustering.ipynb)

#### 3.2 贝叶斯方法-概率编程与贝叶斯推断
- [1-贝叶斯哲学](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%8E%A8%E6%96%AD%E4%B8%8E%E6%A6%82%E7%8E%87%E7%BC%96%E7%A8%8B/1-%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%93%B2%E5%AD%A6.ipynb)
- [2-进一步PyMC](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%8E%A8%E6%96%AD%E4%B8%8E%E6%A6%82%E7%8E%87%E7%BC%96%E7%A8%8B/2-%E8%BF%9B%E4%B8%80%E6%AD%A5PyMC.ipynb)
- [3-打开MCMC的黑箱](http://nbviewer.jupyter.org/github/lj72808up/ML_Handcraft/blob/master/blog/%E8%B4%9D%E5%8F%B6%E6%96%AF%E6%8E%A8%E6%96%AD%E4%B8%8E%E6%A6%82%E7%8E%87%E7%BC%96%E7%A8%8B/3-%E6%89%93%E5%BC%80MCMC%E7%9A%84%E9%BB%91%E7%AE%B1.ipynb)



四. Tuning Parameters
------------------------------
- [GBDT](blog/GBM_Tuning_Parameters.pdf)




五. Mathematics
------------------------------
#### 第一层
#### 1.1 点集拓扑 
- [点集拓扑-王彦英.avi](http://v.youku.com/v_show/id_XNzM4MjU5ODg=.html?spm=a2h1n.8251843.playList.5~5~A&f=22245870&o=1)
- [点集拓扑讲义-与上面视频配套书籍](https://page72.ctfile.com/fs/1623972-206656801)
- [基础拓扑学-阿姆斯壮](http://www.hejizhan.com/html/res/268.html)

#### 1.2 抽象代数
#### 1.3 高等代数与矩阵
#### 第二层
#### 2.1 数学分析
#### 2.2 实分析
#### 2.3 测度论
#### 第三层
#### 3.1 概率
#### 3.2 泛函分析与积分变换
#### 3.3 ODE
#### 第四层
#### 4.1 随机过程
#### 4.2 PDE
#### 4.3 随机微分方程
