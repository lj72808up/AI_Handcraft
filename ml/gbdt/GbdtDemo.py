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


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = getData()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV

    clf = GradientBoostingClassifier(random_state=1,n_estimators=100,max_depth=10)
    # param_test1 = {'n_estimators':[200],'max_depth':[3,7,9]}
    # clf = gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(random_state=10,),
    #                               param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)


    from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix

    accuracy = accuracy_score(y_test, predict)
    fscore = fbeta_score(y_test, predict, beta=0.5)
    print "Final accuracy score on the validation data: {:.4f}".format(accuracy)
    print "F-score on validation data: {:.4f}".format(fbeta_score(y_test, predict, beta=0.5))
    print confusion_matrix(y_test, predict)

    from sklearn import tree
    import graphviz
    dot_data = tree.export_graphviz(clf, out_file="out1.file")
    graph = graphviz.Source(dot_data)
    graph.render("iris")
