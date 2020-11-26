from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')

X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c=test.drop(['ID'],axis=1).values

# X_train_c1 = train.drop(['ID','CLASS'],axis=1).values
# y_train_c = train['CLASS'].values
# X_test_c1=train.drop(['ID'],axis=1).values
# X_train_c = min_max_scaler.fit_transform(X_train_c1)
# X_test_c= min_max_scaler.fit_transform(X_test_c1)

X_train, X_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=520)
mlp_clf__tuned_parameters = {"hidden_layer_sizes": [(120, 60, 30,15),(120, 60, 30), (100,), (100, 40), (120, 60), (100, 50,25,10,5), (50, 50)],
                             "solver": ['adam', 'sgd', 'lbfgs'],
                             "max_iter": [50000],
                             "verbose": [True],
                             # "activation": ['identity', 'logistic', 'tanh', 'relu'],
                             # "tol":[1e-5,1e-6,1e-4]
                             }

# clf = MLPClassifier()
# estimator = GridSearchCV(clf, mlp_clf__tuned_parameters, n_jobs=6)
# estimator.fit(X_train, y_train)
# print(estimator.get_params().keys())
# print(estimator.best_params_)
# print(estimator.best_score_)

# print(clf)
# clf.fit(X_train,y_train)
# pred=clf.predict(X_test)
# print("scoreMLP=",clf.score(X_test,y_test))

clf1=MLPClassifier()
clf1.fit(X_train,y_train)
print("scoreMLP=",clf1.score(X_test,y_test))
y1 = clf1.predict(X_test)
y_result = clf1.predict(X_test_c)
print("y1", y1, "length of y1", len(y1))
print("scorek3=", clf1.score(X_test, y_test))

id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': y_result})
# DataFrame为二维表
df.to_csv("output/baseline_MLP1.csv", index=False)  # 数据导出