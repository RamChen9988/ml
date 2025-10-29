from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
print("开始训练线性SVM分类器...")

# 创建线性可分数据集
X, y = make_classification(n_samples=100, n_features=2, 
                          n_redundant=0, n_informative=2,
                          n_clusters_per_class=1, random_state=42)

# 创建线性SVM分类器
# 试验不同的C值
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X, y)

print("训练完成！")
print(f"支持向量数量：{len(linear_svm.support_vectors_)}")
print(f"模型参数：w={linear_svm.coef_}, b={linear_svm.intercept_}")

# 绘制决策边界和支持向量
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
plt.scatter(linear_svm.support_vectors_[:, 0], 
            linear_svm.support_vectors_[:, 1], 
            s=100, facecolors='none', edgecolors='k')
# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# 创建网格来评估模型
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = linear_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 绘制决策边界和间隔
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
           alpha=0.5, linestyles=['--', '-', '--'])
plt.title('线性SVM分类结果')
plt.show()