import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=420)




s= np.arange(0, 2, 0.1)
dt_clf__tuned_parameters = {
                            "criterion": ['gini','entropy'],
                             "max_depth": [i for i in range(4, 30)],
                             # "min_weight_fraction_leaf ": s,
                             "max_features": ['auto','log2','sqrt'],
                            # "class_weight":['balanced']
                             # "activation": ['identity', 'logistic', 'tanh', 'relu'],
                             # "tol":[1e-5,1e-6,1e-4]
                             }



dt_clf=tree.DecisionTreeClassifier()
# dt_clf.fit(X_train_c,y_train_c)
estimator = GridSearchCV(dt_clf, dt_clf__tuned_parameters, n_jobs=2)
estimator.fit(x_train, y_train)
print(estimator.get_params().keys())
print(estimator.best_params_)
print(estimator.best_score_)

# y_result = dt_clf.predict(X_test_c)
# id_ = range(210, 314)
# df = pd.DataFrame({'ID': id_, 'CLASS': y_result})
# # DataFrame为二维表
# df.to_csv("output/baseline_Dt.csv", index=False)  # 数据导出
