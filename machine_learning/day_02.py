from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_boston, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

from data.common_data import DATA_DIR


def li():

    li = load_iris()

    # 获取特征值
    print("获取特征值")
    print(li.data)

    # 获取目标值
    print("获取目标值")
    print(li.target)
    print(li.DESCR)

    # 注意返回值 包含训练集 train x_train y_train  和测试集  x_test y_test
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)
    print("训练集的特征值和目标值：", x_train, y_train)
    print("测试集的特征值和目标值：", x_test, y_test)

    news = fetch_20newsgroups(subset='all')
    print(news.data)
    print(news.target)

    # lb = load_boston()
    # # 获取特征值
    # print("获取特征值")
    # print(lb.data)
    #
    # # 获取目标值
    # print("获取目标值")
    # print(lb.target)
    # print(lb.DESCR)


def decision():
    """
    决策树对泰坦尼克号预测
    :return:
    """
    # 获取数据
    # titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    titan = pd.read_csv(os.path.join(DATA_DIR, "titan/titan.txt"))

    # 处理数据 找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]

    print(x)

    # 缺失值处理 填入平均值
    x['age'].fillna(x['age'].mean(), inplace=True)

    y = titan[['survived']]

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理 特征工程  特征是类别的信息需要 one_hot编码
    dic = DictVectorizer(sparse=False)
    x_train = dic.fit_transform(x_train.to_dict(orient="records"))

    print(dic.get_feature_names())

    x_test = dic.transform(x_test.to_dict(orient="records"))

    print(x_train)

    # # 用决策树进行预测
    # dec = DecisionTreeClassifier(max_depth=5)
    # dec.fit(x_train, y_train)
    #
    # # 预测准确率
    # print("预测准确率为: ", dec.score(x_test, y_test))
    #
    # # 导出决策树的结果
    #
    # export_graphviz(dec, out_file=os.path.join(DATA_DIR, 'tree.dot'),
    #                 feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])

    # 随机森林进行预测 超参数调优
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=10)

    gc.fit(x_train, y_train)

    print("预测准确率为：", gc.score(x_test, y_test))

    print("查看选择的参数模型:", gc.best_params_)

    return None


if __name__ == '__main__':
    decision()



