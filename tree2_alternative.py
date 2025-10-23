# 决策树回归实战 - 使用内置数据集版本
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

# 使用糖尿病数据集（内置，无需下载）
print("加载糖尿病数据集...")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

print(f"数据集信息: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
print(f"特征名称: {feature_names}")

# 创建回归树
reg_tree = DecisionTreeRegressor(
    max_depth=9,                    # 控制树深度
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
print(f"\n模型评估结果:")
print(f"训练集R²: {train_score:.3f}")
print(f"测试集R²: {test_score:.3f}")

# 显示特征重要性
feature_importance = reg_tree.feature_importances_
print(f"\n特征重要性:")
for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
    print(f"  {name}: {importance:.3f}")

# 预测示例
print(f"\n预测示例:")
print(f"前5个测试样本的真实值: {y_test[:5]}")
print(f"前5个测试样本的预测值: {reg_tree.predict(X_test[:5])}")

# 模型参数信息
print(f"\n决策树参数:")
print(f"树深度: {reg_tree.get_depth()}")
print(f"叶节点数量: {reg_tree.get_n_leaves()}")
