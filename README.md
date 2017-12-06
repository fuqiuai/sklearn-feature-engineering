# kaggle-feature-engineering
使用sklearn做特征工程

[TOC]


## 1. 什么是特征工程？
有这么一句话在业界广泛流传，**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。**那特征工程到底是什么呢？顾名思义，其本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。

<br>特征工程主要分为三部分：
1. **数据预处理** 对应的sklearn包：[sklearn-Processing data](http://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation)
1. **特征选择** 对应的sklearn包： [sklearn-Feature selection](http://scikit-learn.org/stable/modules/feature_selection.html)
1. **降维** 对应的sklearn包： [sklearn-Dimensionality reduction](http://scikit-learn.org/stable/modules/decomposition.html#decompositions)

<br>本文中使用sklearn中的IRIS（鸢尾花）数据集来对特征处理功能进行说明导入IRIS数据集的代码如下：
```
 1 from sklearn.datasets import load_iris
 2 
 3 #导入IRIS数据集
 4 iris = load_iris()
 5 
 6 #特征矩阵
 7 iris.data
 8 
 9 #目标向量
10 iris.target

```


## 2. 数据预处理
通过特征提取，我们能得到未经处理的特征，这时的特征可能有以下问题：

- 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。**无量纲化**可以解决这一问题。
- 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。**二值化**可以解决这一问题。
- 定性特征不能直接使用：通常使用哑编码的方式将定性特征转换为定量特征，假设有N种定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展特征赋值为1，其他扩展特征赋值为0。哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用**哑编码**后的特征可达到非线性的效果。
- 存在缺失值：**填充缺失值**。
- 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的**数据变换**，都能达到非线性的效果。

我们使用sklearn中的preproccessing库来进行数据预处理，可以覆盖以上问题的解决方案。

### 2.1 无量纲化
无量纲化使不同规格的数据转换到同一规格
#### 2.1.1 标准化
将服从正态分布的特征值转换成标准正态分布（对列向量处理），标准化需要计算特征的均值和标准差，公式表达为：
![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113957732-1062097580.png)
<br>使用preproccessing库的StandardScaler类对数据进行标准化的代码如下：
```
1 from sklearn.preprocessing import StandardScaler
2 
3 #标准化，返回值为标准化后的数据
4 StandardScaler().fit_transform(iris.data)
```
#### 2.1.2 区间缩放
#### 2.1.3 归一化

### 2.2 二值化

### 2.3 分类特征编码

### 2.4 缺失值计算

### 2.5 数据变换

## 3. 特征选择


## 4. 降维

