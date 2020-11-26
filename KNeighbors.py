from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('res/train.csv')
test = pd.read_csv('res/test.csv')
X_train_c = train.drop(['ID', 'CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(X_train_c, y_train_c, test_size=0.3, random_state=420)
param_grid = [
    {
        'weights': ["uniform"],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]
best_p = -1
best_score = 0.0
best_k = -2
best_method = ""
best_score = 0.0
best_k = -2
for method in ["uniform","distance"]:
    for k in range(1,15):
        knn_clf = KNeighborsClassifier(n_neighbors=k,weights=method)
        knn_clf.fit(x_train,y_train)
        score = knn_clf.score(x_test,y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method

print("best_k",best_k)
print("best_method",best_method)
print("best_score",best_score)

# y_result = knn_clf.predict(X_test_c)
# id_ = range(210, 314)
# df = pd.DataFrame({'ID': id_, 'CLASS': y_result})
# # DataFrame为二维表
# df.to_csv("output/baseline_k1.csv", index=False)  # 数据导出


# knn_clf = KNeighborsClassifier()
# # n_jobs 代表需要用几个核，传入-1，代表用所有核（我的是双核）verbose 搜索中输出信息
# grid_search = model_selection.GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)
#
clf = MLPClassifier()
estimator = GridSearchCV(clf, mlp_clf__tuned_parameters, n_jobs=6)
estimator.fit(X_train, y_train)
print(grid_search.get_params().keys())
print(grid_search.best_params_)
print(grid_search.best_score_)


# knn_clf = KNeighborsClassifier(weights='distance',p=4,n_neighbors=4)
# knn_clf.fit(x_train, y_train)
# y_result = knn_clf.predict(X_test_c)
# id_ = range(210, 314)
# df = pd.DataFrame({'ID': id_, 'CLASS': y_result})
# # DataFrame为二维表
# df.to_csv("output/baseline_k4.csv", index=False)  # 数据导出