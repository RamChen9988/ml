# -*- coding: utf-8 -*-
"""
逻辑回归实战项目：鸢尾花分类
适用对象：机器学习初学者
作者：AI教师陈辉星
日期：2025年
"""

# 1. 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("逻辑回归分类实战：鸢尾花品种识别")
print("=" * 60)

# 2. 加载和探索数据集
print("\n第一步：加载和探索数据集")
print("-" * 40)

# 加载鸢尾花数据集
iris = load_iris()
print(f"数据集名称: {iris['DESCR'][:100]}...") if iris['DESCR'] else None

# 创建DataFrame以便更好地查看数据
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_type'] = df['target'].apply(lambda x: iris.target_names[x])

print(f"\n数据集形状: {df.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"目标类别: {iris.target_names}")

print("\n数据集前5行:")
print(df.head())

print("\n数据基本统计信息:")
print(df.describe())

print("\n各类别样本数量:")
print(df['flower_type'].value_counts())

# 3. 数据可视化 - 探索特征与类别的关系
print("\n第二步：数据可视化探索")
print("-" * 40)

# 创建特征配对散点图
print("正在生成数据可视化图表...")
plt.figure(figsize=(15, 10))

# 散点图矩阵
colors = ['red', 'blue', 'green']
for i, flower_type in enumerate(iris.target_names):
    plt.scatter(df[df['flower_type'] == flower_type]['sepal length (cm)'],
                df[df['flower_type'] == flower_type]['sepal width (cm)'],
                c=colors[i], label=flower_type, alpha=0.7)

plt.xlabel('花萼长度 (cm)')
plt.ylabel('花萼宽度 (cm)')
plt.title('鸢尾花数据集 - 花萼尺寸分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 特征关系热力图
plt.figure(figsize=(10, 8))
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# 4. 准备数据用于模型训练
print("\n第三步：准备数据用于模型训练")
print("-" * 40)

# 选择特征和目标变量
X = df[iris.feature_names]  # 所有特征
y = df['target']            # 目标类别

print(f"特征数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)  # stratify确保各类别比例一致

print(f"训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")

print("\n训练集中各类别分布:")
print(pd.Series(y_train).value_counts().sort_index())

print("\n测试集中各类别分布:")
print(pd.Series(y_test).value_counts().sort_index())

# 5. 数据标准化（可选但推荐）
print("\n第四步：数据标准化")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("数据标准化完成！")
print("标准化后的数据示例（前5行）:")
print(X_train_scaled[:5])

# 6. 创建和训练逻辑回归模型
print("\n第五步：创建和训练逻辑回归模型")
print("-" * 40)

# 创建逻辑回归模型
# multi_class='multinomial' 用于多分类问题，solver='lbfgs' 是推荐的优化算法
model = LogisticRegression(multi_class='multinomial', 
                          solver='lbfgs',
                          max_iter=1000,  # 增加迭代次数确保收敛
                          random_state=42)

print("开始训练模型...")
model.fit(X_train_scaled, y_train)
print("模型训练完成!")

# 7. 查看模型参数
print("\n第六步：查看模型参数")
print("-" * 40)

print("模型系数（权重）:")
for i, feature in enumerate(iris.feature_names):
    print(f"  {feature}: {model.coef_[:, i]}")

print(f"\n模型截距: {model.intercept_}")

print("\n模型参数解释:")
print("- 系数（权重）表示每个特征对分类决策的影响程度")
print("- 正系数表示特征值增加时，属于该类别的概率增加")
print("- 负系数表示特征值增加时，属于该类别的概率减少")

# 8. 使用模型进行预测
print("\n第七步：使用模型进行预测")
print("-" * 40)

# 对测试集进行预测
y_pred = model.predict(X_test_scaled)

# 预测概率
y_pred_proba = model.predict_proba(X_test_scaled)

# 创建结果对比DataFrame
results = pd.DataFrame({
    '实际类别': y_test,
    '实际花种': [iris.target_names[i] for i in y_test],
    '预测类别': y_pred,
    '预测花种': [iris.target_names[i] for i in y_pred],
    '预测正确': y_test == y_pred
})

# 添加每个类别的预测概率
for i, flower_name in enumerate(iris.target_names):
    results[f'{flower_name}概率'] = y_pred_proba[:, i]

print("预测结果对比 (前10个样本):")
print(results.head(10))

# 9. 评估模型性能
print("\n第八步：评估模型性能")
print("-" * 40)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 计算其他评估指标
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数: {f1:.4f}")

print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 10. 可视化评估结果
print("\n第九步：可视化评估结果")
print("-" * 40)

# 创建混淆矩阵可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')

# 创建特征重要性条形图
plt.subplot(1, 2, 2)
feature_importance = np.abs(model.coef_).mean(axis=0)
plt.barh(iris.feature_names, feature_importance)
plt.xlabel('平均特征重要性（系数绝对值）')
plt.title('特征重要性分析')
plt.tight_layout()
plt.show()

# 11. 概率分析可视化
print("\n第十步：概率分析")
print("-" * 40)

# 选择几个样本展示概率分析
sample_indices = [0, 5, 10]  # 选择测试集中的几个样本

plt.figure(figsize=(15, 5))
for i, idx in enumerate(sample_indices):
    plt.subplot(1, 3, i+1)
    sample_proba = y_pred_proba[idx]
    plt.bar(iris.target_names, sample_proba, color=['red', 'blue', 'green'], alpha=0.7)
    plt.ylim(0, 1)
    plt.title(f'样本 {idx} 分类概率\n实际: {iris.target_names[y_test.iloc[idx]]}')
    plt.ylabel('预测概率')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 12. 使用模型进行新预测
print("\n第十一步：使用模型进行新预测")
print("-" * 40)

# 创建一些新的花朵测量数据用于预测
# 格式: [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 类似setosa
    [6.0, 2.7, 4.0, 1.2],  # 类似versicolor
    [6.5, 3.0, 5.5, 2.0]   # 类似virginica
])

# 标准化新数据
new_flowers_scaled = scaler.transform(new_flowers)

# 使用训练好的模型进行预测
new_predictions = model.predict(new_flowers_scaled)
new_probabilities = model.predict_proba(new_flowers_scaled)

print("新花朵品种预测:")
for i, flower in enumerate(new_flowers):
    pred_class = new_predictions[i]
    pred_name = iris.target_names[pred_class]
    probabilities = new_probabilities[i]
    
    print(f"\n花朵 {i+1} 测量值: {flower}")
    print(f"预测品种: {pred_name}")
    print("各类别概率:")
    for j, prob in enumerate(probabilities):
        print(f"  {iris.target_names[j]}: {prob:.4f} ({prob*100:.2f}%)")

# 13. 模型决策边界可视化（使用前两个特征）
print("\n第十二步：决策边界可视化（前两个特征）")
print("-" * 40)

# 为了可视化，我们只使用前两个特征
X_2d = X_train_scaled[:, :2]  # 只使用前两个标准化后的特征
model_2d = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model_2d.fit(X_2d, y_train)

# 创建网格点
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测每个网格点的类别
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap=plt.cm.RdYlBu, 
                     edgecolor='black', s=50)
plt.xlabel('花萼长度（标准化后）')
plt.ylabel('花萼宽度（标准化后）')
plt.title('逻辑回归决策边界（使用前两个特征）')
plt.colorbar(scatter)
plt.show()

# 13. 不同阈值效果对比
print("\n第十三步：不同阈值效果对比")
print("-" * 40)

thresholds = [0.5, 0.3, 0.7]
for threshold in thresholds:
    y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
    accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
    precision_threshold = precision_score(y_test, y_pred_threshold, average='weighted', zero_division=0)
    recall_threshold = recall_score(y_test, y_pred_threshold, average='weighted', zero_division=0)
    f1_threshold = f1_score(y_test, y_pred_threshold, average='weighted', zero_division=0)
    
    print(f"\n阈值: {threshold}")
    print(f"  准确率: {accuracy_threshold:.4f} ({accuracy_threshold*100:.2f}%)")
    print(f"  精确率: {precision_threshold:.4f}")
    print(f"  召回率: {recall_threshold:.4f}")
    print(f"  F1分数: {f1_threshold:.4f}")

# 14. 总结
print("\n" + "=" * 60)
print("项目总结")
print("=" * 60)

print("""
本项目完成了以下内容:
1.  加载和探索了鸢尾花数据集
2.  可视化分析了特征与花种类别的关系
3.  准备了训练数据和测试数据，并进行了标准化
4.  创建并训练了多类别逻辑回归模型
5.  理解了模型参数的意义
6.  使用模型进行了花种预测和概率分析
7.  全面评估了模型的性能（准确率、精确率、召回率、F1分数）
8.  可视化了混淆矩阵和特征重要性
9.  分析了模型的预测概率
10. 可视化了决策边界

关键知识点:
- 逻辑回归用于分类问题，输出的是概率
- 多分类问题可以使用multinomial逻辑回归
- 数据标准化通常能提高模型性能
- 评估分类模型需要多个指标，不能只看准确率
- 模型的系数反映了特征对分类决策的重要性

接下来可以尝试:
- 尝试使用不同的特征组合
- 调整模型参数（如正则化强度C）
- 尝试其他分类算法（如决策树、SVM）并比较结果
- 处理更复杂的真实世界分类问题
""")

# 保存模型（可选）
# import joblib
# joblib.dump(model, 'logistic_regression_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("模型和标准化器已保存")
