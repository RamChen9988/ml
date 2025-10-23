# 决策树回归实战
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import warnings

# 尝试加载加州房价数据集，如果失败则使用替代数据集
try:
    print("正在加载加州房价数据集...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    print(f"成功加载加州房价数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    
except Exception as e:
    print(f"加载加州房价数据集失败: {e}")
    print("使用生成的回归数据集作为替代...")
    
    # 生成一个类似的回归数据集
    X, y = make_regression(
        n_samples=1000, 
        n_features=8, 
        noise=0.1, 
        random_state=42
    )
    
    # 创建有意义的特征名称
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    print(f"使用生成的替代数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

# 创建回归树
reg_tree = DecisionTreeRegressor(
    max_depth=4,                    # 控制树深度
    min_samples_split=20,           # 分裂最小样本数
    min_samples_leaf=10,            # 叶节点最小样本数
    random_state=42
)

# 分割数据并训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_tree.fit(X_train, y_train)

# 评估模型
train_score = reg_tree.score(X_train, y_train)
test_score = reg_tree.score(X_test, y_test)
print(f"训练集R²: {train_score:.3f}, 测试集R²: {test_score:.3f}")

# 显示特征重要性
feature_importance = reg_tree.feature_importances_
print("\n特征重要性:")
for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"  {name}: {importance:.3f}")

# 预测示例
print(f"\n预测示例:")
print(f"前5个测试样本的真实值: {y_test[:5]}")
print(f"前5个测试样本的预测值: {reg_tree.predict(X_test[:5])}")
