# -*- coding: utf-8 -*-
# @author : wanglei
# @date : 2021/2/19 1:47 PM
# @description :
import numpy as np

"""
感应器对象
"""


class Perceptron(object):
    """
    该方法为感应器的初始化方法
    eta:学习速率
    n_iter:学习次数（迭代次数）
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    """
    该方法为模型训练的方法
    shape[0]返回该矩阵有几行
    shape[1]返回该矩阵有几列
    在这个例子中X.shape[1]=2
    np.zeros(1 + X.shape[1])是一个1行3列的元素都为零的列表
    """

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # 初始化一个权重和阈值的列表，初始值为0
        self.errors_ = []  # 用来记录每一次迭代全样本的错误预测次数
        for _ in range(self.n_iter):  # 进行多次预测样本
            errors = 0  # 用来记录本次预测的全样本的错误次数
            for xi, target in zip(X, y):  # 遍历这个样本集和实际结果集
                update = self.eta * (
                            target - self.predict(xi))  # 用实际结果值减掉预测结果值如果该值为0，表示预测正确，如果不为0则乘上学习速率，获取的值就是本次权重、阈值需要更新的值
                self.w_[1:] += update * xi  # 如果预测正确，则update为0，那么权重本次就无需改变，否则，增加
                self.w_[0] += update  # 如果预测正确，则update为0，那么阈值本次就无需改变，否则，增加
                errors += int(update != 0.0)  # 预测错误就记录一次错误数
            self.errors_.append(errors)  # 将所有的样本数据预测完成后,将本次的预测错误的次数放到error_这个列表中
        return self

    """
    该方法为将一个样本的属性值进行处理的方法
    X=array([[1,2,3,4],[5,6,7,8],...])
    self.w_[1:]=array([0,0,0,0])
    根据api：dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    np.dot(X,self.w_[1:])=array([[0],[0],...])【将每一个属性乘上权重再将每一个样本的每个属性值进行求和】
    self.w_[0]=array([[0]])获取阈值
    """

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """
    该方法为一个样本的预测结果输出方法
    numpy.where(condition[, x, y])
    就是一个三目运算，满足条件就输出x，否则输出y
    """

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

import pandas as pd
"""
读取数据源
"""
df = pd.read_csv("/Users/a1/Downloads/iris.data", header=None)
print(df.tail())  # 打印后几行
y = df.iloc[0:100, 4].values  # 取前100行数据的第4列，类标这一列，前100行就两类
print(y)
y = np.where(y == 'Iris-setosa', -1, 1)  # 将类标这一列的文本表示替换成数字表示，就分了两类
X = df.iloc[0:100, [0, 2]].values  # 获取前100行的第0列和第2列，即花瓣宽度和花萼宽度
print(X)

"""
对模型进行训练，查看训练时每次迭代的错误数量
"""
ppn= Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)