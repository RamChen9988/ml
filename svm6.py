import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 生成极端不平衡的线性可分数据
np.random.seed(42)

# 类别0（密集类）：数量多、分布集中
n_dense = 200  # 样本多
X0 = np.random.normal(loc=[1, 1], scale=0.5, size=(n_dense, 2))  # 分布密集
y0 = np.zeros(n_dense)

# 类别1（稀疏类）：数量少、分布稀疏
n_sparse = 20  # 样本少
X1 = np.random.normal(loc=[5, 5], scale=1.2, size=(n_sparse, 2))  # 分布稀疏
y1 = np.ones(n_sparse)

# 合并数据
X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# 2. 训练线性模型
svm = SVC(kernel='linear')  # 线性SVM
lr = LogisticRegression()   # 逻辑回归
svm.fit(X, y)
lr.fit(X, y)

# 3. 可视化决策边界
def plot_boundary(model, X, y, title, is_svm=False):
    # 网格范围
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    
    # 绘制样本点
    plt.scatter(X0[:,0], X0[:,1], c='blue', label='类别0（密集）', alpha=0.6)
    plt.scatter(X1[:,0], X1[:,1], c='orange', label='类别1（稀疏）', alpha=0.6)
    
    # 绘制决策边界
    w = model.coef_[0]
    b = model.intercept_[0]
    x_line = np.linspace(x_min, x_max)
    y_line = (-w[0]/w[1])*x_line - b/w[1]
    plt.plot(x_line, y_line, 'r-', linewidth=3, label='决策边界')
    
    # SVM标记支持向量
    if is_svm:
        plt.scatter(X[model.support_,0], X[model.support_,1],
                   s=150, facecolors='none', edgecolors='green',
                   label='支持向量')
    
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()

# 对比绘图
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_boundary(svm, X, y, 'SVM（线性核）：优先最大间隔', is_svm=True)

plt.subplot(122)
plot_boundary(lr, X, y, '逻辑回归：受样本数量影响', is_svm=False)

plt.tight_layout()
plt.show()