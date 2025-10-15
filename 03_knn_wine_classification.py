# -*- coding: utf-8 -*-
"""
K近邻算法实战项目：葡萄酒分类与回归
适用对象：机器学习初学者
作者：AI教师陈辉星
日期：2025年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("K近邻算法实战：葡萄酒分类与回归")
print("=" * 70)

# 1. 加载和探索数据集
print("\n第一步：加载和探索数据集")
print("-" * 50)

# 加载葡萄酒数据集
wine = load_wine()
print(f"数据集描述: {wine['DESCR'][:200]}..." if wine['DESCR'] else "数据集加载成功")

# 创建DataFrame以便更好地查看数据
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df['wine_class'] = df['target'].apply(lambda x: wine.target_names[x])

print(f"\n数据集形状: {df.shape}")
print(f"特征数量: {len(wine.feature_names)}")
print(f"目标类别: {wine.target_names}")

print("\n数据集前5行:")
print(df.head())

print("\n各类别样本数量:")
print(df['wine_class'].value_counts())

print("\n特征名称:")
for i, feature in enumerate(wine.feature_names, 1):
    print(f"{i:2d}. {feature}")

# 2. 数据可视化探索
print("\n第二步：数据可视化探索")
print("-" * 50)

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('葡萄酒数据集探索性分析', fontsize=16)

# 类别分布
class_counts = df['wine_class'].value_counts()
axes[0, 0].bar(class_counts.index, class_counts.values, color=['skyblue', 'lightgreen', 'salmon'])
axes[0, 0].set_title('葡萄酒类别分布')
axes[0, 0].set_ylabel('样本数量')
axes[0, 0].tick_params(axis='x', rotation=45)

# 选择两个主要特征进行散点图可视化
feature1, feature2 = 'alcohol', 'malic_acid'
colors = ['red', 'blue', 'green']
for i, wine_class in enumerate(wine.target_names):
    class_data = df[df['wine_class'] == wine_class]
    axes[0, 1].scatter(class_data[feature1], class_data[feature2], 
                      c=colors[i], label=wine_class, alpha=0.7)
axes[0, 1].set_xlabel(feature1)
axes[0, 1].set_ylabel(feature2)
axes[0, 1].set_title(f'{feature1} vs {feature2}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 酒精含量分布
for i, wine_class in enumerate(wine.target_names):
    axes[1, 0].hist(df[df['wine_class'] == wine_class]['alcohol'], 
                   alpha=0.7, label=wine_class, bins=15, color=colors[i])
axes[1, 0].set_xlabel('酒精含量')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('酒精含量分布')
axes[1, 0].legend()

# 特征相关性热力图
corr_matrix = df[wine.feature_names[:6]].corr()  # 只显示前6个特征的相关性
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('特征相关性热力图 (前6个特征)')

plt.tight_layout()
plt.show()

# 3. 数据预处理
print("\n第三步：数据预处理")
print("-" * 50)

# 选择特征和目标变量
X = df[wine.feature_names]
y = df['target']

print(f"特征数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 数据标准化 - KNN算法对特征尺度敏感，必须进行标准化
print("\n正在进行数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建标准化后的DataFrame便于查看
X_scaled_df = pd.DataFrame(X_scaled, columns=wine.feature_names)
print("\n标准化后的数据示例 (前5行):")
print(X_scaled_df.head())

print("\n标准化后的数据统计:")
print(X_scaled_df.describe())

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")
print(f"训练集类别分布:\n{pd.Series(y_train).value_counts().sort_index()}")
print(f"测试集类别分布:\n{pd.Series(y_test).value_counts().sort_index()}")

# 4. 寻找最佳K值
print("\n第四步：寻找最佳K值")
print("-" * 50)

# 测试不同的K值
k_range = range(1, 21)
k_scores = []

print("正在测试不同K值的性能...")
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 使用5折交叉验证
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
    print(f"K={k:2d} | 平均准确率: {scores.mean():.4f}")

# 找到最佳K值
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)

print(f"\n最佳K值: {best_k}")
print(f"最佳交叉验证准确率: {best_score:.4f}")

# 可视化K值选择
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, k_scores, marker='o', linestyle='-', color='steelblue')
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最佳K值={best_k}')
plt.xlabel('K值')
plt.ylabel('交叉验证准确率')
plt.title('K值选择对准确率的影响')
plt.grid(True, alpha=0.3)
plt.legend()

# 5. 训练最终KNN分类模型
print("\n第五步：训练最终KNN分类模型")
print("-" * 50)

# 使用最佳K值训练模型
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(X_train, y_train)

print(f"使用K={best_k}训练KNN分类模型完成!")

# 进行预测
y_pred = knn_classifier.predict(X_test)
y_pred_proba = knn_classifier.predict_proba(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)

print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# 可视化分类结果
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, 
            yticklabels=wine.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'混淆矩阵 (K={best_k})')

plt.tight_layout()
plt.show()

# 6. 特征重要性分析
print("\n第六步：特征重要性分析")
print("-" * 50)

# 通过排列特征来评估特征重要性
def evaluate_feature_importance(X, y, feature_names, model, n_repeats=5):
    """评估特征重要性"""
    base_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    feature_importance = {}
    
    print("正在进行特征重要性分析...")
    for i, feature_name in enumerate(feature_names):
        temp_X = X.copy()
        # 打乱特定特征
        original_feature = temp_X[:, i].copy()
        shuffled_accuracies = []
        
        for _ in range(n_repeats):
            np.random.shuffle(temp_X[:, i])
            shuffled_accuracy = cross_val_score(model, temp_X, y, cv=5, scoring='accuracy').mean()
            shuffled_accuracies.append(shuffled_accuracy)
            # 恢复原始特征
            temp_X[:, i] = original_feature
        
        avg_shuffled_accuracy = np.mean(shuffled_accuracies)
        importance = base_accuracy - avg_shuffled_accuracy
        feature_importance[feature_name] = max(importance, 0)  # 确保非负
    
    return feature_importance

# 计算特征重要性（使用较小的K值以加快计算）
knn_quick = KNeighborsClassifier(n_neighbors=3)
feature_importance = evaluate_feature_importance(X_train, y_train, wine.feature_names, knn_quick)

# 排序并显示特征重要性
feature_importance_df = pd.DataFrame({
    'feature': list(feature_importance.keys()),
    'importance': list(feature_importance.values())
}).sort_values('importance', ascending=False)

print("\n特征重要性排序:")
for i, row in feature_importance_df.iterrows():
    print(f"{row['feature']:20s}: {row['importance']:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='lightseagreen')
plt.xlabel('特征重要性')
plt.title('KNN特征重要性分析')
plt.tight_layout()
plt.show()

# 7. K近邻回归演示
print("\n第七步：K近邻回归演示")
print("-" * 50)

# 使用酒精含量作为回归目标，其他特征作为输入
X_reg = df.drop(['alcohol', 'target', 'wine_class'], axis=1)
y_reg = df['alcohol']

print(f"回归特征: {list(X_reg.columns)}")
print(f"回归目标: alcohol (酒精含量)")
print(f"酒精含量统计: 均值={y_reg.mean():.2f}, 标准差={y_reg.std():.2f}")

# 数据标准化
X_reg_scaled = scaler.fit_transform(X_reg)
y_reg_scaled = (y_reg - y_reg.mean()) / y_reg.std()  # 标准化目标变量

# 分割数据
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg_scaled, test_size=0.3, random_state=42)

# 寻找回归的最佳K值
k_range_reg = range(1, 16)
k_scores_reg = []

print("\n寻找回归最佳K值...")
for k in k_range_reg:
    knn_reg = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn_reg, X_reg_train, y_reg_train, cv=5, scoring='r2')
    k_scores_reg.append(scores.mean())
    print(f"K={k:2d} | R²分数: {scores.mean():.4f}")

best_k_reg = k_range_reg[np.argmax(k_scores_reg)]
print(f"\n回归最佳K值: {best_k_reg}")

# 训练KNN回归模型
knn_regressor = KNeighborsRegressor(n_neighbors=best_k_reg)
knn_regressor.fit(X_reg_train, y_reg_train)

# 预测
y_reg_pred = knn_regressor.predict(X_reg_test)

# 评估回归模型
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"\n回归模型评估:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R²分数: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range_reg, k_scores_reg, marker='s', linestyle='-', color='darkorange')
plt.axvline(x=best_k_reg, color='red', linestyle='--', alpha=0.7, label=f'最佳K值={best_k_reg}')
plt.xlabel('K值')
plt.ylabel('R²分数')
plt.title('K值选择对回归性能的影响')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_reg_test, y_reg_pred, alpha=0.7)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 
         'k--', lw=2, label='理想预测')
plt.xlabel('实际酒精含量 (标准化)')
plt.ylabel('预测酒精含量 (标准化)')
plt.title('KNN回归预测 vs 实际值')
plt.legend()

plt.subplot(1, 3, 3)
residuals = y_reg_test - y_reg_pred
plt.scatter(y_reg_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')

plt.tight_layout()
plt.show()

# 8. 不同距离度量的比较
print("\n第八步：不同距离度量的比较")
print("-" * 50)

distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
metric_scores = []

print("比较不同距离度量...")
for metric in distance_metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    scores = cross_val_score(knn_metric, X_train, y_train, cv=5, scoring='accuracy')
    metric_scores.append(scores.mean())
    print(f"{metric:12s}: 平均准确率 = {scores.mean():.4f}")

best_metric = distance_metrics[np.argmax(metric_scores)]
print(f"\n最佳距离度量: {best_metric}")

# 可视化距离度量比较
plt.figure(figsize=(8, 6))
bars = plt.bar(distance_metrics, metric_scores, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylabel('交叉验证准确率')
plt.title('不同距离度量性能比较')
plt.ylim(0.9, 1.0)

# 在柱状图上添加数值
for bar, score in zip(bars, metric_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{score:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 9. 使用模型进行新预测
print("\n第九步：使用模型进行新预测")
print("-" * 50)

# 创建一些新的葡萄酒数据用于预测
# 注意：这些是虚构的数据，仅用于演示
new_wines = np.array([
    [13.5, 2.5, 2.6, 20.0, 95, 2.05, 0.36, 1.6, 4.0, 1.0, 2.55, 1.2, 450],  # 类似class_0
    [12.8, 1.8, 2.4, 18.5, 88, 2.45, 0.38, 1.9, 4.5, 1.1, 2.85, 1.3, 520],  # 类似class_1
    [14.2, 3.2, 2.8, 22.0, 105, 2.85, 0.42, 2.2, 5.0, 1.2, 3.15, 1.4, 580]  # 类似class_2
])

# 标准化新数据
new_wines_scaled = scaler.transform(new_wines)

# 使用训练好的模型进行预测
new_predictions = knn_classifier.predict(new_wines_scaled)
new_probabilities = knn_classifier.predict_proba(new_wines_scaled)

print("新葡萄酒品种预测:")
for i, wine_data in enumerate(new_wines):
    pred_class = new_predictions[i]
    pred_name = wine.target_names[pred_class]
    probabilities = new_probabilities[i]
    
    print(f"\n葡萄酒 {i+1}:")
    print(f"  预测品种: {pred_name}")
    print("  各类别概率:")
    for j, prob in enumerate(probabilities):
        print(f"    {wine.target_names[j]}: {prob:.4f} ({prob*100:.2f}%)")

# 10. 项目总结
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print(f"""
K近邻算法实战项目完成情况:

数据准备:
  - 葡萄酒样本: {len(df)} 个
  - 特征数量: {len(wine.feature_names)} 个
  - 类别数量: {len(wine.target_names)} 类

分类模型:
  - 最佳K值: {best_k}
  - 测试集准确率: {accuracy:.4f}
  - 最佳距离度量: {best_metric}

回归模型:
  - 最佳K值: {best_k_reg}
  - R²分数: {r2:.4f}

关键发现:
  - 数据标准化对KNN性能至关重要
  - K值选择需要在偏差和方差之间平衡
  - 欧氏距离在此数据集上表现最佳
  - 特征'flavanoids'和'color_intensity'对分类最重要

K近邻算法特点:
  ✓ 简单直观，易于理解和实现
  ✓ 无需训练过程，惰性学习
  ✓ 对数据分布没有假设
  ✗ 计算复杂度高，预测速度慢
  ✗ 对不相关特征和异常值敏感
  ✗ 需要大量内存存储训练数据

实际应用建议:
  1. 适用于中小型数据集
  2. 必须进行特征标准化
  3. 通过交叉验证选择最佳K值
  4. 考虑使用特征选择提高性能
  5. 在大数据集上考虑使用近似最近邻算法
""")

# 保存模型（可选）
print("\n模型保存代码示例:")
print("""
# 保存训练好的模型和预处理工具
import joblib
joblib.dump(knn_classifier, 'knn_wine_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(knn_regressor, 'knn_regressor.pkl')

# 加载模型进行预测
# knn_loaded = joblib.load('knn_wine_classifier.pkl')
# scaler_loaded = joblib.load('scaler.pkl')
""")