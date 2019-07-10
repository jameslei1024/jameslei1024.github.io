> 西瓜书学习笔记
>
> James
>
> 2019-05-28

# 支持向量机(Support Vector Machine,SVM)
## 划分超平面、间隔和支持向量
给定样本集 $D=\{(\textbf{x}_1,y_1),(\textbf{x}_2,y_2),\dots,(\textbf{x}_n,y_n)\},y_i\in\{-1,+1\}$ ,分类学习最基本的想法就是基于训练集在样本空间中找到一个划分超平面将不同类别分开。

在样本空间中，划分超平面可以表示为 $\textbf{w}^T\textbf{x}+b=0$。其中 $\textbf{w}=(w_1;w_2;\dots;w_n)$ 为法向量，决定超平面的方向；$b$为偏移量，决定超平面与原点的距离。

假设超平面能够将训练样本正确分类，即对于任意一个$(\textbf{x}_i,y_i)\in D$，$y_i\times(\textbf{w}^T\textbf{x}_i+b)\gt 0$。

通过缩放，我们能够使
$$
\textbf{w}^T\textbf{x}_i+b\ge+1，y_i=+1\\
\textbf{w}^T\textbf{x}_i+b\le-1，y_i=-1
$$
其中，使等号成立的样本称为**支持向量**(support vector)。

两个异类支持向量到平面的距离和称为**间隔**(margin)，$\gamma=\frac{2}{||\textbf{w}||}$。

![svm.png](http://www.coxdocs.org/lib/exe/fetch.php?media=perseus:user:activities:matrixprocessing:learning:svm.png)

## 支持向量机的基本型
支持向量机所想要解决的问题即为找到具有最大间隔的划分超平面。即
$$
\begin{aligned}
\max_{\textbf{w},b}\quad&\frac{2}{||\textbf{w}||}
\\s.t.\quad& y_i(\textbf{w}^T\textbf{x}_i+b)\geq1,\quad i=1,2,\dots,n
\end{aligned}
$$
上式亦可重写为
$$
\begin{aligned}
\min_{\textbf{w},b}\quad&\frac{1}{2}||\textbf{w}||^2
\\s.t.\quad& y_i(\textbf{w}^T\textbf{x}_i+b)\geq1,\quad i=1,2,\dots,n
\end{aligned}
$$
我们可以使用*拉格朗日乘子法*获得上式的对偶问题为，
$$
\begin{aligned}
\max_\textbf{a}\quad&\sum_{i=1}^{n}a_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^na_ia_jy_iy_j\textbf{x}_i^T\textbf{x}_j\\
s.t.\quad&\sum_{i=1}^na_iy_i=0,\\
&a_i\geq0,\quad i=1,2,..,n
\end{aligned}
$$
其*KKT条件*为
$$
\begin{aligned}
&a_i\geq0\\
&y_if(\textbf{x}_i)-1\geq0\\
&a_i(y_if(\textbf{x}_i)-1)=0
\end{aligned}
$$
解出 $\textbf{a}$ 后,求出 $\textbf{w}$ 与 $b$ 即可得到模型 $f(\textbf{x})=\textbf{w}^T\textbf{x}+b=\sum_{i=1}^na_iy_i\textbf{x}^T_i\textbf{x}+b$

## SMO优化算法

SMO算法(Sequential Minimal Optimization)是常用的快速求解SVM问题的方法。

SMO的基本思路是先固定 $a_i$ 以外的所有参数，然后求 $a_i$ 上的极值。由于存在约束 $\sum_{i=1}^na_iy_i=0$,若固定 $a_i$ 之外的其他变量，则 $a_i$ 可由其他变量导出。于是SMO每次选择两个变量 $a_i$ 和 $a_j$，并固定其他参数。

> SMO算法流程：
> 
> 1. 初始化参数
> 2. 选取一对需要更新的变量$a_i$和$a_j$
> 3. 固定$a_i$和$a_j$以外的参数,求解对偶问题获得更新后的$a_i$和$a_j$
> 4. 重复 2. 和 3. 直至收敛

## 核函数

SVM所能解决的问题是线性可分的问题，即存在一个划分超平面能够将训练样本正确分类。

然而对于一些实际问题，原始样本空间内可能无法找到一个能够将所有样本正确分类的划分超平面。然而我们知道，如果原始空间是有限维，即属性数有限，那么一定存在一个**高维特征空间**使样本可分。

令 $\phi$ 为将向量映射到高维空间的函数，则 $\phi(\textbf{x})$ 为将 $\textbf{x}$ 映射到高维的向量。此时划分超平面为$f(x)=\textbf{w}^T\phi(\textbf{x})+b$，而SVM要解决的对偶问题为：
$$
\begin{aligned}
\max_\textbf{a}\quad&\sum_{i=1}^{n}a_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^na_ia_jy_iy_j\phi(\textbf{x}_i)^T\phi(\textbf{x}_j)\\
s.t.\quad&\sum_{i=1}^na_iy_i=0,\\
&a_i\geq0,\quad i=1,2,..,n
\end{aligned}
$$
由于映射后的特征空间维度很高，计算其内积十分困难，因此我们提出了核函数 $\kappa(\textbf{x}_i,\textbf{x}_j)=\langle\phi(\textbf{x}_i),\phi(\textbf{x}_j)\rangle=\phi(\textbf{x}_i)^T\phi(\textbf{x}_j)$。

此时，我们需要解决的问题变成了

$$
\begin{aligned}
\max_\textbf{a}\quad&\sum_{i=1}^{n}a_i-\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^na_ia_jy_iy_j\kappa(\textbf{x}_i,\textbf{x}_j)\\
s.t.\quad&\sum_{i=1}^na_iy_i=0,\\
&a_i\geq0,\quad i=1,2,..,n
\end{aligned}
$$
求解后即可得到$f(\textbf{x})=\textbf{w}^T\phi(\textbf{x})+b=\sum_{i=1}^na_iy_i\phi(\textbf{x})^T\phi(\textbf{x})+b=\sum_{i=1}^na_iy_i\kappa(\textbf{x}_i,\textbf{x}_j)+b$

不合适的核函数将样本映射到了不合适的特征空间，可能会导致性能不佳。

常用的核函数有：
- 线性核
  $$\kappa(\textbf{x}_i,\textbf{x}_j)=\textbf{x}_i^T\textbf{x}_j$$
- 多项式核
  $$\kappa(\textbf{x}_i,\textbf{x}_j)=(\textbf{x}_i^T\textbf{x}_j)^d$$
  $d\ge1$为多项式的次数
- 高斯核
  $$\kappa(\textbf{x}_i,\textbf{x}_j)=\exp(-\frac{||\textbf{x}_i-\textbf{x}_j||^2}{2\sigma^2})$$
  $\sigma\gt0$为高斯核的带宽(width)
- 拉普拉斯核
  $$\kappa(\textbf{x}_i,\textbf{x}_j)=\exp(-\frac{||\textbf{x}_i-\textbf{x}_j||}{\sigma})$$
  $\sigma\gt0$
- Sigmoid核
  $$\kappa(\textbf{x}_i,\textbf{x}_j)=\tanh(\beta\textbf{x}_i^T\textbf{x}_j+\theta)$$
  $\tanh$为双曲正切函数，$\beta\gt0,\theta\lt0$

除此之外，核函数还可以通过函数组合得到。

## 软间隔

完全线性可分的情况在现实中出现的几率不大，所以我们应该允许支持向量机在一些样本上出错，因此我们引入了**软间隔**这一概念。

此时，我们的优化目标为
$$
\min_{\textbf{w},b}\quad\frac{1}{2}||\textbf{w}||^2+C\sum_{i=1}^nl_{0/1}(y_i(\textbf{w}^T\textbf{x}_i+b)-1) 
$$
其中$C\gt0$是一个常数，$l_{0/1}$是0/1损失函数，其表示的是误分类样本的个数。

<img style="display:block;margin:0 auto" src="https://latex.codecogs.com/gif.latex?l_{0/1}=\left\{\begin{matrix}&space;1,&space;&&space;\text{if&space;}z<0&space;\\&space;0,&space;&&space;\text{otherwise}&space;\end{matrix}\right." title="l_{0/1}=\left\{\begin{matrix} 1, & \text{if }z<0 \\ 0, & \text{otherwise} \end{matrix}\right." />

其他常用的损失函数：
- hinge损失
  $$
  l_{hinge}(z)=\max(0,1-z)
  $$
- 指数损失(exponential loss)
  $$
  l_{exp}(z)=\exp(-z)
  $$
- 对率损失(logistic loss)
  $$
  l_{log}(z)=\log(1+\exp(-z))
  $$

![常见的损失函数](https://images2018.cnblogs.com/blog/1188231/201804/1188231-20180427193102542-1527793912.png)

**没看懂,TBC**


## 支持向量回归(Support Vector Regression,SVR)

**TBC**