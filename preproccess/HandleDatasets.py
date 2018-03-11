# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def getDataset():
    data = pd.read_csv("../datasets/census.csv")
    #print(data.shape)

    # 将数据切分成特征和对应的标签
    # income 列是我们需要的标签，记录一个人的年收入是否高于50K。 因此我们应该把他从数据中剥离出来，单独存放。
    income_raw = data['income']
    features_raw = data.drop('income',axis=1)

    # 对于高度倾斜分布的特征如'capital-gain'和'capital-loss'，常见的做法是对数据施加一个对数转换，
    # 将数据转换成对数，这样非常大和非常小的值不会对学习算法产生负面的影响。并且使用对数变换显著降低了由于异常值所造成的数据范围异常。
    # 但是在应用这个变换时必须小心：因为0的对数是没有定义的，所以我们必须先将数据处理成一个比0稍微大一点的数以成功完成对数转换。
    skewed = ['capital-gain','capital-loss']
    features_raw[skewed] = data[skewed].apply(lambda x:np.log(x+1)) # numpy操作dataframe

    # 规一化数字特征 : 让每个特征转变为(x-min)/(max-min)   0~1
    # 除了对于高度倾斜的特征施加转换，对数值特征施加一些形式的缩放通常会是一个好的习惯。在数据上面施加一个缩放并不会改变数据分布的形式
    # （比如上面说的'capital-gain' or 'capital-loss'）；但是，规一化保证了每一个特征在使用监督学习器的时候能够被平等的对待。
    # 注意一旦使用了缩放，观察数据的原始形式不再具有它本来的意义了，就像下面的例子展示的。
    # 运行下面的代码单元来规一化每一个数字特征。我们将使用sklearn.preprocessing.MinMaxScaler来完成这个任务。
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numerical = ['age','education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_raw[numerical] = scaler.fit_transform(data[numerical])

    # 独热编码 : 很多回归分类模型, 是基于欧氏距离来进行的, 而有些数据的feature, 是文字进行分类而不是数字类型.
    # eg : 某个feature成绩的值为{优,良,中}. 这三个文字分类在转换成数字时, 不能简单地变成{1,2,3}, 否则'优'和'中'的欧氏距离变成根号2,
    #      大于'优'和'良'欧氏距离1, 而分类feature的分类值应该相似度一致(欧式距离一致), 因此, 需要进行独热编码, 把一个feature拆成3个feature.
    #      把优,良,中 提升成feature, 取消成绩feature. 原来成绩为优的, 现在的"优"feature变成1 . (此时优,良,中三个点变成(1,0,0),(0,1,0),(0,0,1)), 欧式距离都是1
    # TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
    features = pd.get_dummies(features_raw)
    # TODO：将'income_raw'编码成数字值
    income = pd.get_dummies(income_raw).iloc[:,1:]  # 只取收入大于50k作为输出字段
    #print("income: %s"%income)
    # 打印经过独热编码之后的特征数量
    encoded = list(features.columns)
    print "{} total features after one-hot encoding.".format(len(encoded))

    # 混洗和切分数据
    # 现在所有的 类别变量 已被转换成数值特征，而且所有的数值特征已被规一化。和我们一般情况下做的一样，我们现在将数据（包括特征和它们的标签）切分成训练和测试集。
    # 其中80%的数据将用于训练和20%的数据用于测试。然后再进一步把训练数据分为训练集和验证集，用来选择和优化模型。
    # 导入 train_test_split
    from sklearn.model_selection import train_test_split
    # 将'features'和'income'数据切分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,stratify = income)
    # 将'X_train'和'y_train'进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,stratify = y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test