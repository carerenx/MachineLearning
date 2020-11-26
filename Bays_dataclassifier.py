import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler
train=pd.read_csv('res/train.csv')
test=pd.read_csv('res/test.csv')
X_train_c=train.drop(['ID','CLASS'],axis=1).values
y_train_c=train['CLASS'].values
X_test_c=test.drop(['ID'],axis=1).values
x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=420)
x_train_t=MinMaxScaler(feature_range=(0, 10)).fit_transform(x_train)
x_test_t=MinMaxScaler(feature_range=(0, 10) ).fit_transform(x_test)

mub=GaussianNB()
mub.fit(X_train_c,y_train_c)
y_predict=mub.predict(X_test_c)
id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': y_predict})
df.to_csv("output/baseline_bays.csv", index=False)  # 数据导出
pred1 = GaussianNB().fit(x_train, y_train)
pred2 = MultinomialNB().fit(x_train_t, y_train)
pred3 = BernoulliNB().fit(x_train, y_train)
score1 = pred1.score(x_test, y_test)
score2 = pred2.score(x_test, y_test)
score3 = pred3.score(x_test, y_test)
print(score1)
print(score2)
print(score3)
y_pred = pred1.predict(x_test)
prob = pred1.predict_proba(x_test)


