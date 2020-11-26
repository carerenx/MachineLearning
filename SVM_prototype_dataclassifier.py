# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from pandas.plotting import table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR

train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')
# 分离数据集
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values  # drop id and class
y_train_c = train['CLASS'].values  # class is result,train contain 242 columns
# X大写是因为X是数据集，y小写是因为y是特征值
X_test_c = test.drop(['ID'], axis=1).values

# do not include class(result),test contain 241 cloumns,axis=0 vertical axis=1 horizontal
nfold= 10
kf = KFold(n_splits=nfold, shuffle=True, random_state=55)
# k交叉验证，n_splits:divide into sevral part,shuffle:not in order,random_state:contral random model
# 把数据分为k份，前k-1份用于训练，最后一份用于验证
prediction1 = np.zeros((len(X_test_c),))
prediction2 = np.zeros((len(X_train_c)//nfold,))
prediction3 = np.zeros((len(X_test_c),))
print("(len(X_test_c)=",(len(X_test_c)))
print("(len(X_test_c)//nfold=",(len(X_test_c)//nfold))
# creat array that size of x_test_c,len:length

i = 0
for train_index, valid_index in kf.split(X_train_c, y_train_c):
    # 多迭代变量的for循环，分开循环train_index循环168次，valid_index循环42次
    # valid有效的。
    # train_index为168个索引值,valid_index为42个索引值，他们都是数组
    print("\nFold {}".format(i + 1))
    X_train, label_train = X_train_c[train_index], y_train_c[train_index]
    X_valid, label_valid = X_train_c[valid_index], y_train_c[valid_index]
    # X_valid是42个元组，label_valid是42个结果(1/5)
    clf = SVR(kernel='rbf', C=2.72, gamma='scale')
    # kernel：核函数类型，C:惩罚因子：C表征你有多么重视离群点，C越大越重视，越不想丢掉它们
    # gamma是’rbf’，’poly’和’sigmoid’的核系数且gamma的值必须大于0。
    # 随着gamma的增大，存在对于测试集分类效果差而对训练分类效果好的情况，
    # 并且容易泛化误差出现过拟合。
    clf.fit(X_train, label_train)
    # 用训练数据拟合分类器模型
    x1 = clf.predict(X_valid)
    y1 = clf.predict(X_test_c)

    prediction1 += y1 / nfold
    prediction2 += x1 / nfold
    prediction3 += ((y1)) / nfold
    print("prediction1",prediction1)
    print("prediction3",prediction3)
    # 双括号里面的那个括号是元组
    print('Accuracy:\t', accuracy_score(label_valid, np.round(prediction2)))
    precision = precision_score(label_valid, np.round(prediction2))
    print('Precision:\t', precision)
    recall = recall_score(label_valid, np.round(prediction2))
    print('Recall:\t', recall)
    print('f1 score:\t', f1_score(label_valid, np.round(prediction2)))

    i += 1
result1 = np.round(prediction1)
# round四舍五入
id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
# DataFrame为二维表
df.to_csv("output/baseline_123.csv", index=False)  # 数据导出


