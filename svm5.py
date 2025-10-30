import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 生成非线性可分的模拟数据（两类数据，边界呈环形）
np.random.seed(42)  # 固定随机种子，保证结果可复现

# 生成内层点（类别0）
inner_points = 100
radius_inner = 5
theta = np.random.uniform(0, 2*np.pi, inner_points)
x1_inner = radius_inner * np.cos(theta) + np.random.normal(0, 0.5, inner_points)
x2_inner = radius_inner * np.sin(theta) + np.random.normal(0, 0.5, inner_points)
X_inner = np.column_stack((x1_inner, x2_inner))
y_inner = np.zeros(inner_points)

# 生成外层点（类别1）
outer_points = 100
radius_outer = 10
x1_outer = radius_outer * np.cos(theta) + np.random.normal(0, 0.5, outer_points)
x2_outer = radius_outer * np.sin(theta) + np.random.normal(0, 0.5, outer_points)
X_outer = np.column_stack((x1_outer, x2_outer))
y_outer = np.ones(outer_points)

# 合并数据
X = np.vstack((X_inner, X_outer))
y = np.hstack((y_inner, y_outer))

# 2. 训练模型
# SVM（RBF核，擅长处理非线性边界）
svm = SVC(kernel='rbf', gamma=0.1)
svm.fit(X, y)

# 逻辑回归（仅能学习线性边界）
lr = LogisticRegression()
lr.fit(X, y)

# 3. 评估准确率
svm_pred = svm.predict(X)
lr_pred = lr.predict(X)

print(f"SVM准确率: {accuracy_score(y, svm_pred):.4f}")
print(f"逻辑回归准确率: {accuracy_score(y, lr_pred):.4f}")

# 4. 可视化决策边界
def plot_boundary(model, X, y, title):
    # 生成网格点
    h = 0.1
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')

# 对比绘图
plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_boundary(svm, X, y, f'SVM (准确率: {accuracy_score(y, svm_pred):.4f})')

plt.subplot(122)
plot_boundary(lr, X, y, f'逻辑回归 (准确率: {accuracy_score(y, lr_pred):.4f})')

plt.tight_layout()
plt.show()