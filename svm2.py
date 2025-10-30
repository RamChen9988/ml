from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# 网格搜索
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X, y)

print("最佳参数：", grid.best_params_)
print("最佳得分：", grid.best_score_)

# 使用最佳参数创建最终模型
best_svm = grid.best_estimator_