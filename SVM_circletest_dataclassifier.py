# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from pandas.plotting import table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from sklearn.svm import SVR

train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')
# 分离数据集
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
X_test_c = test.drop(['ID'], axis=1).values
y_train_c = train['CLASS'].values
prediction1 = np.zeros((len(X_test_c),))

clf = SVR(kernel='rbf', C=1e3, gamma='scale')
print("1e3=", 1e3)
clf.fit(X_train_c, y_train_c)

y1 = clf.predict(X_test_c)
# print(y1)

result1 = np.round(y1)
# round四舍五入
id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
# DataFrame为二维表
df.to_csv("output/baseline2.csv", index=False)  # 数据导出

j = 0
m = 0
t = 0
a=np.arange(1, 100000,0.1)
print(a)
for i in a:
    x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=25043)
    clf1 = SVR(kernel='rbf', C=i, gamma='scale')
    pred1 = clf1.fit(x_train, y_train)
    y_result = clf1.predict(x_test)
    j = pred1.score(x_test, y_test)
    if j > m:
        m = j
        t = i
    print("i=",i,"score1=", j)


print("scoreMax=", m, "i=", t)
# x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=59)
# clf1 = SVR(kernel='rbf', C=1e3, gamma='scale')
# pred1 = clf1.fit(x_train, y_train)
# y_result = clf1.predict(x_test)
# print("score1=", pred1.score(x_test, y_test))
