import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA  # 用于降维可视化

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 加载数据
wine = load_wine()
X = wine.data  # 特征：葡萄酒化学成分（13维）
y = wine.target  # 标签：3种产地（0,1,2）

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()  # 标准化特征（SVM对特征尺度敏感）
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 训练模型
# SVM模型（RBF核函数，适合非线性分类）
svm = SVC(kernel='rbf', gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# 逻辑回归模型
lr = LogisticRegression(max_iter=1000, random_state=42)  # 增加迭代次数确保收敛
lr.fit(X_train_scaled, y_train)

# 4. 模型评估
svm_pred = svm.predict(X_test_scaled)
lr_pred = lr.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"SVM准确率: {svm_acc:.4f}")
print(f"逻辑回归准确率: {lr_acc:.4f}")

# 5. 可视化分类结果（用PCA降维到2D）
pca = PCA(n_components=2)  # 将13维特征降为2维以便可视化
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 绘制决策边界的函数
def plot_decision_boundary(model, X, y, title):
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 用模型预测网格点类别
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))  # 反推回原始特征空间预测
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title(title)
    plt.xlabel('PCA特征1')
    plt.ylabel('PCA特征2')

# 绘制两个模型的决策边界
plt.figure(figsize=(12, 5))

plt.subplot(121)
plot_decision_boundary(svm, X_test_pca, y_test, f'SVM (准确率: {svm_acc:.4f})')

plt.subplot(122)
plot_decision_boundary(lr, X_test_pca, y_test, f'逻辑回归 (准确率: {lr_acc:.4f})')

plt.tight_layout()
plt.show()