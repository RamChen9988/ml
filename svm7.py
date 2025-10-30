import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 生成带迷惑性样本的线性可分数据
np.random.seed(42)

# 基础线性可分数据（类别0在左，类别1在右）
n_base = 80
X0_base = np.random.normal(loc=[1, 1], scale=0.6, size=(n_base, 2))  # 类别0
X1_base = np.random.normal(loc=[5, 5], scale=0.6, size=(n_base, 2))  # 类别1

# 添加迷惑性样本（少量类别0混入类别1区域，干扰模型）
n_tricky = 10
X0_tricky = np.random.normal(loc=[4, 4], scale=0.4, size=(n_tricky, 2))  # 类别0的"叛徒"

# 合并数据
X0 = np.vstack([X0_base, X0_tricky])  # 类别0：基础样本+迷惑样本
X1 = X1_base                          # 类别1：纯净样本
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(len(X0)), np.ones(len(X1))])

# 2. 训练模型
svm = SVC(kernel='linear')
lr = LogisticRegression()
svm.fit(X, y)
lr.fit(X, y)

# 3. 计算准确率（这里会产生明显差异）
svm_pred = svm.predict(X)
lr_pred = lr.predict(X)
svm_acc = accuracy_score(y, svm_pred)
lr_acc = accuracy_score(y, lr_pred)

# 4. 可视化
def plot_boundary(model, X, y, title, is_svm=False):
    # 网格范围
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # 绘制样本（区分基础样本和迷惑样本）
    plt.scatter(X0_base[:,0], X0_base[:,1], c='blue', label='类别0（正常）', alpha=0.6)
    plt.scatter(X0_tricky[:,0], X0_tricky[:,1], c='blue', marker='x', s=100, label='类别0（迷惑）')
    plt.scatter(X1[:,0], X1[:,1], c='orange', label='类别1', alpha=0.6)
    
    # 决策边界
    w = model.coef_[0]
    b = model.intercept_[0]
    x_line = np.linspace(x_min, x_max)
    y_line = (-w[0]/w[1])*x_line - b/w[1]
    plt.plot(x_line, y_line, 'r-', linewidth=3, label='决策边界')
    
    # SVM支持向量
    if is_svm:
        plt.scatter(X[model.support_,0], X[model.support_,1],
                   s=150, facecolors='none', edgecolors='green', label='支持向量')
    
    plt.title(f'{title}（准确率：{accuracy_score(y, model.predict(X)):.4f}）')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()

# 对比绘图
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_boundary(svm, X, y, 'SVM（线性核）', is_svm=True)

plt.subplot(122)
plot_boundary(lr, X, y, '逻辑回归', is_svm=False)

plt.tight_layout()
plt.show()