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


def gbdtTrain(X_train, y_train, n):
    from sklearn.ensemble import GradientBoostingClassifier
    # 降低学习率,增加基学习器的个数, 来构造更健壮的模型
    # clf = GradientBoostingClassifier(random_state=1, n_estimators=n,learning_rate=0.2, max_depth=5,verbose=0)
    from sklearn.model_selection import GridSearchCV
    rateList = [(i*1.0)/100 for i in range(5, 40,2)]
    param_test1 = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 6, 7], 'learning_rate': rateList}
    clf = gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(random_state=10, subsample=0.8),
                                  param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    clf = clf.fit(X_train, y_train)
    print clf.best_estimator_
    # print "features weight:%s" % clf.feature_importances_
    return clf


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
    clf = gbdtTrain(X_train, y_train, n)
    # clf = randomForest(X_train,y_train,n)
    evaluate(clf, y_test)
    tree_visualization(clf)
