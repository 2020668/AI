from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba
import numpy as np

# vector = CountVectorizer()

# 调用fit_transform输入并转换数据
# res = vector.fit_transform(["life is short, I like python", "life is long, I dislike python"])
#
# print(vector.get_feature_names())
#
# print(res.toarray())


# 对字典进行特征抽取
def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dictv = DictVectorizer()   # 默认是sparse矩阵  sparse=False  ndarray数组

    # 调用fit_transform
    data = dictv.fit_transform([{"city": "北京", "temperature": 100},
                                {"city": "上海", "temperature": 60},
                                {"city": "深圳", "temperature": 30}
                                ])
    print(dictv.get_feature_names())
    print(dictv.inverse_transform(data))
    print(data)
    return None


def countvec():
    """
    文本特征值化
    :return:
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["life is short, I like python", "life is too long, I dislike python"])
    print(cv.get_feature_names())   # 统计文章中所有的词（单个字母除外），去重，列表
    print(data.toarray())   # 对每篇文章 在词的列表里面进行统计每个词

    return None


def cut_word():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝大部分死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年前发出的，这样当我们在看宇宙时，我们是在看它的过去")

    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它，了解事物真正含义的秘密取决于如何将其与我们了解的事物相联系。")

    # 转换成list
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 将列表转换成字符串
    c1 = " ".join(content1)
    c2 = " ".join(content2)
    c3 = " ".join(content3)

    return c1, c2, c3


def hansvec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cut_word()
    print(c1, c2, c3)

    cv = CountVectorizer()

    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def tfidfvec():
    """
    中文特征值化
    :return: None
    """
    c1, c2, c3 = cut_word()
    print(c1, c2, c3)

    tf = TfidfVectorizer()

    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return: None
    """
    mm = MinMaxScaler(feature_range=(2, 3))     # 数据缩放
    # 默认0到1
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)

    return None


def stand():
    """
    标准化缩放
    :return: None
    """
    std = StandardScaler()
    data = std.fit_transform([[1, -1, 3], [2, 4, 2], [4, 6,-1]])
    print(data)

    return None


def im():
    """
    缺失值处理
    :return: None
    """
    # Nan nan
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    data = imp.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)


def var():
    """
    特征选择 删除低方差的特征
    :return: None
    """
    var = VarianceThreshold(threshold=0)    # 默认0 在0到10范围内  依据实际效果来定
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


def pca():
    """
    主成分分析 进行数据降维
    :return: None
    """
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)

    return None


if __name__ == '__main__':
    pca()
