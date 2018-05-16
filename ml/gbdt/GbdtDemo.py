# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def getData():
    data = pd.read_csv("../../datasets/census.csv")
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # 独热编码,只取收入大于50k作为输出字段
    income = pd.get_dummies(income_raw).iloc[:, 1:]

    # 处理取值范围很大的特征
    skewed = ['capital-gain', 'capital-loss']
    features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
    features = pd.get_dummies(features_raw)

    # PCA降低独热编码后的特征数量
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=30)
    # features = pca.fit_transform(features)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0,
                                                        stratify=income)
    # 将'X_train'和'y_train'进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test


def gbdtTrain(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.externals.joblib import Memory
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from multiprocessing import cpu_count

    from sklearn.feature_selection import SelectKBest, chi2
    # 降低GBDT学习率,增加基学习器的个数, 来构造更健壮的模型
    # pipeline中存在特征处理的操作,可以通过缓存的手段避免模型调参时反复转换特征
    pipe = Pipeline(steps=[("step1_reduceDim", None),
                           ("step2_clf", GradientBoostingClassifier(random_state=10, subsample=0.8))],
                    memory = Memory(cachedir="./tmp",verbose=0))
    from sklearn.decomposition import PCA, NMF
    n_estimators= [50]
    max_depth = [3]
    n_features = [8, 10]
    # "step1_reduceDim__n_components":表示pipeline中step1_reduceDim这一步的n_components参数
    # 定义两个流水线,每个流水线中都有一组可选参数. 网格搜索所有流水线的所有参数组合
    param1 = {"step1_reduceDim": [PCA(iterated_power=7), NMF()],
              "step1_reduceDim__n_components":n_features,
              "step2_clf__n_estimators":n_estimators,
              "step2_clf__max_depth":max_depth}
    param2 = {"step1_reduceDim": [SelectKBest(chi2)],
              "step1_reduceDim__k": n_features,
              "step2_clf__n_estimators":n_estimators,
              "step2_clf__max_depth":max_depth}
    paramGrid = [param1,param2]
    cpuNumber = cpu_count()
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import make_scorer
    # 3折交叉验证,开启cpuNum个线程并行网格搜索
    gridClf = GridSearchCV(pipe,cv=3,
                           n_jobs=cpuNumber,
                           param_grid=paramGrid,
                           scoring=make_scorer(fbeta_score,beta=0.5))
    import time
    start = time.time()
    gridClf = gridClf.fit(X_train, y_train)
    stop = time.time()
    print "training cost %s s" % int(stop - start)
    print gridClf.best_estimator_
    print "====================================="
    print "cv_results: %s" % gridClf.cv_results_['mean_test_score'] # 输出的均值为f1-score评分
    # 打印网格搜索的所有参数组合机器得分
    # print gridClf.grid_scores_
    print "====================================="
    print "features weight:"
    # 从网格中获取最优的pipeline,再从中获取gbdt分类器的属性权重
    print gridClf.best_estimator_.named_steps['step2_clf'].feature_importances_
    from shutil import rmtree
    rmtree("./tmp")
    return gridClf


def randomForest(X_train, y_train, n):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n)
    clf = clf.fit(X_train, y_train)
    print "randomforest weight:%s" % clf.feature_importances_
    return clf


def evaluate(clf, y_test):
    from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix
    predict = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predict)
    fscore = fbeta_score(y_test, predict, beta=0.5)
    print "Final accuracy score on the validation data: {:.4f}".format(accuracy)
    print "F-score on validation data: {:.4f}".format(fscore)
    print confusion_matrix(y_test, predict)


def tree_visualization(clf):
    import graphviz
    # 查看GBDT的第一棵树
    from sklearn import tree
    print "estimator number: %s" % str(clf.estimators_.shape)
    # dot_data = tree.export_graphviz(clf.estimators_[0][0], out_file=None,
    #                                 filled=True,
    #                                 rounded=True,
    #                                 special_characters=True,
    #                                 proportion=True,
    #                                 )
    # graph = graphviz.Source(dot_data)
    # graph.render("out.file")


if __name__ == "__main__":
    n = 104
    X_train, X_val, X_test, y_train, y_val, y_test = getData()
    clf = gbdtTrain(X_train, y_train)
    # clf = randomForest(X_train,y_train,n)
    evaluate(clf, y_test)
    # tree_visualization(clf)
