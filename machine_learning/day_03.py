from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.externals import joblib
import pandas as pd
import numpy as np


def my_linear():
    """
    线性回归预测波士顿房价
    :return:
    """
    # 获取数据
    lb = load_boston()

    # 分割数据到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)

    # 进行标准化处理 特征值和目标值 实例化两个标准化API
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1, 1))   # 一维数组转换为二维数组
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测
    # 正规方程求解方式预测结果
    # lr = LinearRegression()
    # lr.fit(x_train, y_train)
    # print(lr.coef_)

    # 保存训练好的模型
    # joblib.dump(lr, '../data/model/test.pkl')

    # 使用保存的模型预测结果
    model = joblib.load('../data/model/test.pkl')
    y_predict = std_y.inverse_transform(model.predict(x_test))
    print("保存的模型预测的结果:", y_predict)

    # 预测测试集的房价
    # y_predict = lr.predict(x_test)
    # y_lr_predict = std_y.inverse_transform(lr.predict(x_test))     # 从小数型数据转换成标准化之前的数据
    # print("正规方程预测测试集每个房子的房价:", y_lr_predict)
    #
    # print("正规方程的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    # 梯度下降进行房价预测
    # sgd = SGDRegressor()
    # sgd.fit(x_train, y_train)
    #
    # print(sgd.coef_)
    #
    # # 预测测试集的房价
    # y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    #
    # print("梯度下降预测测试集每个房子的房价:", y_sgd_predict)
    #
    # print("梯度下降的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
    #
    # # 岭回归进行房价预测
    # rd = Ridge(alpha=1.0)
    # rd.fit(x_train, y_train)
    #
    # print(rd.coef_)
    #
    # # 预测测试集的房价
    # y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    #
    # print("岭回归预测测试集每个房子的房价:", y_rd_predict)
    #
    # print("梯岭回归的均方误差:", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return


def logistic():
    """
    逻辑回归做二分类进行癌症预测(根据细胞的属性特征)
    :return: None
    """
    # 由于数据没有title 先构造title
    column = ['Sample code number',
              'Clump Thickness',
              'Uniformity of Cell Size',
              'Uniformity of Cell Shape',
              'Marginal Adhesion',
              'Single Epithelial Cell Size',
              'Bare Nuclei',
              'Bland Chromatin',
              'Normal Nucleoli',
              'Mitoses',
              'Class'
              ]
    data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column)
    print(data)

    # 处理缺失值
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(data[column[1:10]], data[column[10]], test_size=0.25)

    # 进行标准化
    std = StandardScaler()

    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 逻辑回归预测
    lg = LogisticRegression()

    lg.fit(x_train, y_train)
    print(lg.coef_)
    y_predict = lg.predict(x_test)
    print('准确率:', lg.score(x_test, y_test))
    print('召回率:', classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    return None


if __name__ == '__main__':
    logistic()

