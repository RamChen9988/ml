from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import joblib

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
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X, y)

print("训练完成！")
print(f"支持向量数量：{len(linear_svm.support_vectors_)}")
print(f"模型参数：w={linear_svm.coef_}, b={linear_svm.intercept_}")

# 保存模型
joblib.dump(linear_svm, 'linear_svm_model.pkl')
print("模型已保存为 'linear_svm_model.pkl'")

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

# 示例：加载保存的模型并进行预测
print("\n--- 加载保存的模型 ---")
try:
    # 加载保存的模型
    loaded_model = joblib.load('linear_svm_model.pkl')
    print("模型加载成功！")
    
    # 使用加载的模型进行预测
    test_sample = X[:15]  # 使用前15个样本作为测试
    predictions = loaded_model.predict(test_sample)
    print(f"测试样本预测结果: {predictions}")
    print(f"实际标签: {y[:15]}")
    
    # 验证模型参数是否一致
    print(f"加载模型的支持向量数量: {len(loaded_model.support_vectors_)}")
    print(f"加载模型的参数: w={loaded_model.coef_}, b={loaded_model.intercept_}")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 5))
    
    # 子图1：预测结果可视化
    plt.subplot(1, 2, 1)
    colors = ['red' if pred == 0 else 'blue' for pred in predictions]
    plt.scatter(range(len(predictions)), predictions, c=colors, s=100, alpha=0.7)
    plt.plot(range(len(predictions)), predictions, 'k--', alpha=0.5)
    plt.xlabel('样本索引')
    plt.ylabel('预测类别')
    plt.title('测试样本预测结果')
    plt.yticks([0, 1])
    plt.grid(True, alpha=0.3)
    
    # 子图2：预测与实际对比
    plt.subplot(1, 2, 2)
    x_pos = np.arange(len(predictions))
    width = 0.35
    
    plt.bar(x_pos - width/2, predictions, width, label='预测', alpha=0.7, color='orange')
    plt.bar(x_pos + width/2, y[:15], width, label='实际', alpha=0.7, color='green')
    
    plt.xlabel('样本索引')
    plt.ylabel('类别')
    plt.title('预测 vs 实际标签')
    plt.legend()
    plt.xticks(x_pos)
    plt.yticks([0, 1])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算准确率
    accuracy = np.mean(predictions == y[:15])
    print(f"\n预测准确率: {accuracy:.2%}")
    
except FileNotFoundError:
    print("模型文件未找到，请确保模型已保存")
except Exception as e:
    print(f"加载模型时出错: {e}")
