import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 准确率
import dataprocess
import pandas as pd

data=dataprocess.Dataprocess
print(data.x_train)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0,
    'max_depth': 5,
    'lambda': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}
num_rounds = 300

plst = list(params.items())
dtrain = xgb.DMatrix(data.X_train_c,data.y_train_c)
clf=xgb.train(plst,dtrain,num_rounds)

# 对测试集进行预测
# dtest = xgb.DMatrix(data.x_test)
# y_pred = clf.predict(dtest)

# 计算准确率
# accuracy = accuracy_score(data.y_test, y_pred)
# print('accuarcy:%.2f%%' % (accuracy * 100))
# print ("best best_ntree_limit",clf.)

# 显示重要特征
# plot_importance(clf)
# plt.show()

clf1=XGBClassifier(
    silent=0,  # 设置成1则没有运行信息输出，最好是设置为0，是否在运行升级时打印消息
    # nthread = 4  # CPU 线程数 默认最大
    learning_rate=0.2,  # 如同学习率
    min_child_weight=1,
    # 这个参数默认为1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
    # 假设h在0.01附近，min_child_weight为1 意味着叶子节点中最少需要包含100个样本
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    max_depth=5,  # 构建树的深度，越大越容易过拟合
    gamma=0.2,  # 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守，一般0.1 0.2这样子
    subsample=0.8,  # 随机采样训练样本，训练实例的子采样比
    max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计
    colsample_bytree=0.8,  # 生成树时进行的列采样
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
    # reg_alpha=0, # L1正则项参数
    # scale_pos_weight =1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛，平衡正负权重
    # objective = 'multi:softmax', # 多分类问题，指定学习任务和响应的学习目标
    # num_class = 10,  # 类别数，多分类与multisoftmax并用
    n_estimators=200,  # 树的个数
    seed=1000,  # 随机种子
    # eval_metric ='auc'
    num_rounds = 300,
)
clf1.fit(data.x_train,data.y_train)
y_pred1=clf1.predict(data.x_test)
# 计算准确率
accuracy1 = accuracy_score(data.y_test, y_pred1)
print('accuarcy:%.2f%%' % (accuracy1 * 100))
X_test_d=xgb.DMatrix(data.X_test_c)

y_result = clf.predict(X_test_d)
id_ = range(210, 314)
df = pd.DataFrame({'ID': id_, 'CLASS': y_result})
# DataFrame为二维表
df.to_csv("output/baseline_XGBoost2.csv", index=False)  # 数据导出