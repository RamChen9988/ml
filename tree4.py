# 决策树回归实战 - 特征重要性分析
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

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
#==========================================================================================
# 获取特征重要性
feature_importance = reg_tree.feature_importances_

# 创建重要性 DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n特征重要性排序:")
print(importance_df)

# 可视化
plt.figure(figsize=(10, 6))
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('特征重要性')
plt.title('决策树特征重要性分析')
plt.tight_layout()
plt.show()

# 额外分析：显示最重要的特征
print(f"\n最重要的特征: {importance_df.iloc[0]['feature']} (重要性: {importance_df.iloc[0]['importance']:.3f})")

# 显示特征重要性统计
print(f"\n特征重要性统计:")
print(f"平均重要性: {importance_df['importance'].mean():.3f}")
print(f"重要性标准差: {importance_df['importance'].std():.3f}")
print(f"零重要性特征数量: {(importance_df['importance'] == 0).sum()}")
