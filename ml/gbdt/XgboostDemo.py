# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xgboost as xgb


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

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0,
                                                        stratify=income)
    # 将'X_train'和'y_train'进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test


def xgboost_train(X_train, y_train):
    # To load a numpy array into DMatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic',
             'nthread': 4, 'eval_metric': 'auc'}
    # Specify validations set to watch performance
    bst = xgb.train(param,dtrain, num_round)

    # predict
    dtest = xgb.DMatrix(X_test)
    predict = bst.predict(dtest)
    predict[predict>0.5] = 1
    predict[predict<=0.5] = 0

    return bst,predict

# Xgboost + logisticregression
def xgboostAndLogistic(X_train, y_train):
    # To load a numpy array into DMatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic',
             'nthread': 4, 'eval_metric': 'auc'}
    # Specify validations set to watch performance

    bst = xgb.train(param,dtrain, num_round)
    # predict leaf index
    dtest = xgb.DMatrix(X_train)
    leafindex = bst.predict(dtest, ntree_limit=num_round, pred_leaf=True)

    print "xgboost train success"
    # one hot encoding
    features = pd.DataFrame(leafindex,columns=[i for i in range(0,num_round)])
    features = pd.get_dummies(features)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=10)
    clf.fit(features,y_train)
    # evaluate(clf.predict(features),y_train)
    return bst,clf

def predictXgboostAndLogistic(bst,clf):
    dtest = xgb.DMatrix(X_test)
    leafindex = bst.predict(dtest, ntree_limit=num_round, pred_leaf=True)
    features = pd.DataFrame(leafindex,columns=[i for i in range(0,num_round)])
    features = pd.get_dummies(features)

    predict = clf.predict(features)

    # dtest = xgb.DMatrix(X_test)
    # predict = bst.predict(dtest)
    # predict[predict>0.5] = 1
    # predict[predict<=0.5] = 0
    # evaluate(predict,y_test)

    return predict

def evaluate(predict,y_test):
    from sklearn.metrics import fbeta_score, accuracy_score, confusion_matrix
    accuracy = accuracy_score(y_test, predict)
    fscore = fbeta_score(y_test, predict, beta=0.5)
    print "Final accuracy score on the validation data: {:.4f}".format(accuracy)
    print "F-score on validation data: {:.4f}".format(fscore)
    print confusion_matrix(y_test, predict)



if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = getData()
    num_round = 300
    bst,predict = xgboost_train(X_train, y_train)
    # bst,clf = xgboostAndLogistic(X_train, y_train)
    # predict = predictXgboostAndLogistic(bst,clf) # Final accuracy score on the validation data: 0.8703,F-score on validation data: 0.7500
    evaluate(predict,y_test)