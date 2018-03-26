# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def showData():
    data_train = pd.read_csv("../datasets/titanic.csv")
    #data.drop(['PassengerId','Survived','Ticket'],axis=1,inplace=True)
    #print data_train.head()

    import matplotlib.pyplot as plt

    # 把年龄分成4个范围,并计算这些范围内的人数
    fig = plt.figure()
    fig.set(alpha=0.2)
    # bins为列表,按元素范围切割. bins为int数字,自动分成n个区间
    pd.cut(data_train["Age"],bins=[0,20,40,60,80]).value_counts().plot("bar")
    plt.show()

    fig = plt.figure()  # 图片大小可在这个设置
    fig.set(alpha=0.2)  # 设定图表颜色alpha参数

    # subplot2grid 在表格里布局图片. (2,3)表格的size为2行3列, (0,0)为图片显示的位置为1行1列
    plt.subplot2grid((2,3),(0,0))
    data_train.Survived.value_counts().plot(kind='bar') # 绘制柱状图
    plt.title(u"Survived number(1)")
    plt.ylabel(u"number")

    # 2行3列表格中, 在第一行第二列作图
    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar") # 柱状图
    plt.ylabel(u"number")
    plt.title(u"People class")

    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived, data_train.Age)  # 绘制散点图 (x:生存与否,y:年龄)
    plt.ylabel(u"age")
    #plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs
    plt.title(u"age survived scatter")


    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # 对等级为1的人的年龄进行kernel desnsity estimate绘制
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"age")
    plt.ylabel(u"density")
    plt.title(u"age distribution of all classes")
    plt.legend((u'first', u'second',u'third'),loc='best') # sets our legend for our graph.


    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"number on port") # 在各个登机口登陆的人数
    plt.ylabel(u"numbers")
    fig.tight_layout(pad=1)  #设置子图间隔
    plt.show()

    # 查看性别对生还的影响
    fig = plt.figure()
    fig.set(alpha=0.2)
    # female幸存者状况, male幸存者状况
    survived_f = data_train.Survived[data_train.Sex=='female'].value_counts()
    survived_m = data_train.Survived[data_train.Sex=="male"].value_counts()
    ssDF = pd.DataFrame({"female":survived_f,"male":survived_m})
    ssDF.plot(kind="bar")
    plt.show()

    # 各性别下, 各舱级别的生还情况
    fig = plt.figure()
    fig.set(alpha=0.2)
    plt.subplot2grid((1,2),(0,0))
    plt.title("male not survived")
    plt.xlabel("class")
    data_train.Pclass[data_train.Survived==0][data_train.Sex=="male"].value_counts().plot(kind="bar")
    plt.subplot2grid((1,2),(0,1))
    plt.title("female not survived")
    plt.xlabel("class")
    data_train.Pclass[data_train.Survived==0][data_train.Sex=="female"].value_counts().plot(kind="bar")
    plt.show()


if __name__=="__main__":
    showData()