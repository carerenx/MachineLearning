# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from pandas.plotting import table
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold,train_test_split
from sklearn.svm import SVR

train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')
# 分离数据集
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
X_test_c = test.drop(['ID'], axis=1).values
y_train_c = train['CLASS'].values
prediction1 = np.zeros((len(X_test_c),))
x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=25043)
clf1 = SVR(kernel='rbf', C=1e3, gamma='scale')
pred1 = clf1.fit(x_train, y_train)
j = pred1.score(x_test, y_test)
print("score(random_state=25043)=", j)
y_result = pred1.predict(X_test_c)
result1 = np.round(y_result)
id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': result1})
df.to_csv("output/baseline3.csv", index=False)  # 数据导出出
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X_train_c, y_train_c):
    print('train_index:', train_index, 'test_index:', test_index)
    X_train, label_train = X_train_c[train_index], y_train_c[train_index]
    X_valid, label_valid = X_train_c[test_index], y_train_c[test_index]
    clf = SVR(kernel='rbf', C=1e3, gamma='scale')
    clf.fit(X_train, label_train)
    y1 = clf.predict(X_test_c)
    prediction1 += y1 / 209
y_result2=np.round(prediction1)
id_=range(210,314)
df=pd.DataFrame({'ID':id_,'CLASS':y_result2})
df.to_csv('output/baseline_loo.csv',index=False)
