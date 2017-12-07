# encoding=utf-8
'''
用sklearn做特征工程，分为三部分：
1.数据预处理
2.特征选择
3.降维
'''

import pandas as pd
import numpy as np
from numpy import vstack, array, nan
from sklearn.datasets import load_iris

from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

if __name__ == '__main__':

    # 导入IRIS数据集
    iris = load_iris()
    features = iris.data
    labels = iris.target

    '''
    1.数据预处理
    '''

    # 1.1 无量纲化：将不同规格的数据转换到同一规格
    # 1.1.1 标准化：将服从正态分布的特征值转换成标准正态分布（对列向量处理）
    # print(np.mean(features, axis=0))
    # print(np.std(features, axis=0))
    features_new = preprocessing.StandardScaler().fit_transform(features)
    # print(np.mean(features_new, axis=0))
    # print(np.std(features_new, axis=0))
    # 1.1.2 区间缩放：将特征值缩放到[0, 1]区间的数据（对列向量处理）
    features_new = preprocessing.MinMaxScaler().fit_transform(features)
    # 1.1.3 归一化：将行向量转化为“单位向量”（对每个样本处理）
    features_new = preprocessing.Normalizer().fit_transform(features)

    # 1.2 对定量特征二值化:设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
    features_new = preprocessing.Binarizer(threshold=3).fit_transform(features)

    # 1.3 对定性（分类）特征编码(也可用pandas.get_dummies函数)
    enc = preprocessing.OneHotEncoder()
    enc.fit([[0, 0, 3],
             [1, 1, 0],
             [0, 2, 1],
             [1, 0, 2]])
    # print(enc.transform([[0, 1, 3]]))
    # print(enc.transform([[0, 1, 3]]).toarray())

    # 1.4 缺失值计算(也可用pandas.fillna函数)
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    features_new = imp.fit_transform(vstack((array([nan, nan, nan, nan]), features)))

    # 1.5 数据变换
    # 1.5.1 基于多项式变换（对行变量处理）
    features_new = preprocessing.PolynomialFeatures().fit_transform(features)
    # 1.5.2 基于自定义函数变换，以log函数为例
    features_new = preprocessing.FunctionTransformer(np.log1p).fit_transform(features)

    '''
    2.特征选择
    '''
    # 2.1 Filter
    # 2.1.1 方差选择法，选择方差大于阈值的特征
    features_new = feature_selection.VarianceThreshold(threshold=0.3).fit_transform(features)
    # 2.1.2 卡方检验,选择K个与标签最相关的特征
    features_new = feature_selection.SelectKBest(feature_selection.chi2, k=3).fit_transform(features, labels)

    # 2.2 Wrapper
    # 2.2.1 递归特征消除法，这里选择逻辑回归作为基模型，n_features_to_select为选择的特征个数
    features_new = feature_selection.RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(features, labels)

    # 2.3 Embedded
    # 2.3.1 基于惩罚项的特征选择法,这里选择带L1惩罚项的逻辑回归作为基模型
    features_new = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(features, labels)
    # 2.3.2 基于树模型的特征选择法,这里选择GBDT模型作为基模型
    features_new = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit_transform(features, labels)

    '''
    3.降维
    '''
    # 3.1 主成分分析法（PCA）,参数n_components为降维后的维数
    features_new = PCA(n_components=2).fit_transform(features)

    # 3.2 线性判别分析法（LDA）,参数n_components为降维后的维数
    features_new = LDA(n_components=2).fit_transform(features, labels)