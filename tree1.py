# 导入必要的库
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 创建决策树分类器
clf = DecisionTreeClassifier(
    max_depth=3,           # 限制树深度
    random_state=42        # 确保结果可重现
)

# 训练模型
clf.fit(X, y)

print("决策树训练完成！")
print(f"树深度：{clf.get_depth()}")
print(f"叶节点数量：{clf.get_n_leaves()}")