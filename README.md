# kaggle-feature-engineering
使用sklearn做特征工程

#### <a href="#1">1. 什么是特征工程？</a>
#### <a href="#2">2. 数据预处理</a>
#### <a href="#3">3. 特征选择</a>
#### <a href="#4">4. 降维</a>

## <a name="1">1. 什么是特征工程？</a>
有这么一句话在业界广泛流传，**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已**。那特征工程到底是什么呢？顾名思义，其本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。

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


## <a name="2">2. 数据预处理</a>
通过特征提取，我们能得到未经处理的特征，这时的特征可能有以下问题：

- 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。**无量纲化**可以解决这一问题。
- 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。**二值化**可以解决这一问题。
- 定性特征不能直接使用：通常使用哑编码的方式将定性特征转换为定量特征，假设有N种定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展特征赋值为1，其他扩展特征赋值为0。哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用**哑编码**后的特征可达到非线性的效果。
- 存在缺失值：**填充缺失值**。
- 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的**数据变换**，都能达到非线性的效果。

我们使用sklearn中的preproccessing库来进行数据预处理，可以覆盖以上问题的解决方案。

### 2.1 无量纲化
无量纲化使不同规格的数据转换到同一规格

#### 2.1.1 标准化（对列向量处理）
将服从正态分布的特征值转换成标准正态分布，标准化需要计算特征的均值和标准差，公式表达为：
<br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113957732-1062097580.png)
<br>使用preproccessing库的StandardScaler类对数据进行标准化的代码如下：
```
1 from sklearn.preprocessing import StandardScaler
2 
3 #标准化，返回值为标准化后的数据
4 StandardScaler().fit_transform(iris.data)
```

#### 2.1.2 区间缩放（对列向量处理）
区间缩放法的思路有多种，常见的一种为利用两个最值进行缩放，公式表达为：
<br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113301013-1555489078.png)
<br>使用preproccessing库的MinMaxScaler类对数据进行区间缩放的代码如下：
```
1 from sklearn.preprocessing import MinMaxScaler
2 
3 #区间缩放，返回值为缩放到[0, 1]区间的数据
4 MinMaxScaler().fit_transform(iris.data)
```

#### 2.1.3 归一化（对行向量处理）
归一化目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。规则为l2的归一化公式如下：
<br>![](http://images2015.cnblogs.com/blog/927391/201607/927391-20160719002904919-1602367496.png)
<br>使用preproccessing库的Normalizer类对数据进行归一化的代码如下：
```
1 from sklearn.preprocessing import Normalizer
2 
3 #归一化，返回值为归一化后的数据
4 Normalizer().fit_transform(iris.data)
```

### 2.2 二值化
定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0，公式表达如下：
<br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502115121216-456946808.png)
<br>使用preproccessing库的Binarizer类对数据进行二值化的代码如下：
```
1 from sklearn.preprocessing import Binarizer
2 
3 #二值化，阈值设置为3，返回值为二值化后的数据
4 Binarizer(threshold=3).fit_transform(iris.data)
```

### 2.3 分类特征编码（对列向量处理）
由于IRIS数据集的特征皆为定量特征，故使用其目标值进行哑编码（实际上是不需要的）。使用preproccessing库的OneHotEncoder类对数据进行哑编码的代码如下：
```
1 from sklearn.preprocessing import OneHotEncoder
2 
3 #哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
4 OneHotEncoder().fit_transform(iris.target.reshape((-1,1)))
```

### 2.4 缺失值计算（对列向量处理）
由于IRIS数据集没有缺失值，故对数据集新增一个样本，4个特征均赋值为NaN，表示数据缺失。使用preproccessing库的Imputer类对数据进行缺失值计算的代码如下：
```
1 from numpy import vstack, array, nan
2 from sklearn.preprocessing import Imputer
3 
4 #缺失值计算，返回值为计算缺失值后的数据
5 #参数missing_value为缺失值的表示形式，默认为NaN
6 #参数strategy为缺失值填充方式，默认为mean（均值）
7 Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
```

### 2.5 数据变换

#### 2.5.1 多项式变换（对行向量处理）
常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的。4个特征，度为2的多项式转换公式如下：
<br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502134944451-270339895.png)
<br>使用preproccessing库的PolynomialFeatures类对数据进行多项式转换的代码如下：
```
1 from sklearn.preprocessing import PolynomialFeatures
2 
3 #多项式转换
4 #参数degree为度，默认值为2
5 PolynomialFeatures().fit_transform(iris.data)
```

#### 2.5.1 自定义变换
基于单变元函数的数据变换可以使用一个统一的方式完成，使用preproccessing库的FunctionTransformer对数据进行对数函数转换的代码如下：
```
1 from numpy import log1p
2 from sklearn.preprocessing import FunctionTransformer
3 
4 #自定义转换函数为对数函数的数据变换
5 #第一个参数是单变元函数
6 FunctionTransformer(log1p).fit_transform(iris.data)
```

### 总结
|类 | 功能 | 说明|
|- | :-: | -: |
|StandardScaler | 无量纲化 | 标准化，基于特征矩阵的列，将特征值转换至服从标准正态分布|
|MinMaxScaler | 无量纲化 | 区间缩放，基于最大最小值，将特征值转换到[0, 1]区间上|
|Normalizer | 归一化 | 基于特征矩阵的行，将样本向量转换为“单位向量”||
|Binarizer | 二值化 | 基于给定阈值，将定量特征按阈值划分|
|OneHotEncoder | 哑编码 | 将定性数据编码为定量数据|
|Imputer | 缺失值计算 | 计算缺失值，缺失值可填充为均值等|
|PolynomialFeatures | 多项式数据转换 | 多项式数据转换|
|FunctionTransformer | 自定义单元数据转换 | 使用单变元的函数来转换数据|

## <a name="3">3. 特征选择</a>


## <a name="4">4. 降维</a>

