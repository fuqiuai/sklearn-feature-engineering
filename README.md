# kaggle-feature-engineering
使用sklearn做特征工程

[TOC]


## 1.什么是特征工程？
有这么一句话在业界广泛流传：**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。**那特征工程到底是什么呢？顾名思义，其本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。
<br>特征工程主要分为三部分：
1. 数据预处理 需引用的sklearnl[sklearn-Processing data](http://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation)
1. 特征选择 [sklearn-Feature selection](http://scikit-learn.org/stable/modules/feature_selection.html)
1. 降维 [sklearn-Dimensionality reduction](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)

## 2. 数据预处理

### 2.1 无量纲化
无量纲化使不同规格的数据转换到同一规格
#### 2.1.1 标准化
将服从正态分布的特征值转换成标准正态分布（对列向量处理），标准化需要计算特征的均值和标准差，公式表达为：
![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113957732-1062097580.png)
#### 2.1.2 区间缩放
#### 2.1.3 归一化

### 2.2 对定量特征二值化

### 2.3 对定性特征编码

### 2.4 缺失值计算


## 3. 特征选择


## 4. 降维

