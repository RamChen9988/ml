# -*- coding: utf-8 -*-
"""
线性回归实战项目：房价预测与模型评估
适用对象：机器学习初学者
作者：AI教师陈辉星
日期：2025年
"""

# 1. 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体显示（确保系统有中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 2. 加载和探索数据集
print("=" * 50)
print("第一步：加载和探索数据集")
print("=" * 50)

# 创建模拟房价数据集（避免网络下载问题）
# 生成1000个样本的模拟数据
np.random.seed(42)  # 确保结果可重现
n_samples = 1000

# 生成特征数据
income = np.random.normal(3.5, 1.5, n_samples)  # 收入中位数（万美元）
house_age = np.random.normal(28, 12, n_samples)  # 房龄中位数（年）
rooms = np.random.normal(5.4, 2.2, n_samples)    # 平均房间数
population = np.random.normal(1425, 1132, n_samples)  # 人口

# 生成房价数据（基于特征的线性组合加上噪声）
house_price = (2.5 * income + 
               0.1 * house_age + 
               0.8 * rooms - 
               0.0001 * population + 
               np.random.normal(0, 0.5, n_samples))

# 创建DataFrame
df = pd.DataFrame({
    'MedInc': income,
    'HouseAge': house_age,
    'AveRooms': rooms,
    'Population': population,
    '房价': house_price
})

print(f"数据集形状: {df.shape}")  # 显示数据集的维度（行数, 列数）
print("\n数据集的前5行:")
print(df.head())  # 显示前5行数据

print("\n数据集的描述性统计:")
print(df.describe())  # 显示数值型列的基本统计信息

# 3. 数据可视化 - 探索特征与房价的关系
print("\n" + "=" * 50)
print("第二步：数据可视化探索")
print("=" * 50)

# 创建2x2的子图来显示主要特征与房价的关系
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('加州房价数据集 - 特征与房价关系可视化', fontsize=16)

# 绘制收入中位数与房价的关系
axes[0, 0].scatter(df['MedInc'], df['房价'], alpha=0.5)
axes[0, 0].set_xlabel('收入中位数 (万美元)')
axes[0, 0].set_ylabel('房价中位数 (万美元)')
axes[0, 0].set_title('收入 vs 房价')

# 绘制房龄中位数与房价的关系
axes[0, 1].scatter(df['HouseAge'], df['房价'], alpha=0.5, color='orange')
axes[0, 1].set_xlabel('房龄中位数 (年)')
axes[0, 1].set_ylabel('房价中位数 (万美元)')
axes[0, 1].set_title('房龄 vs 房价')

# 绘制房间平均数与房价的关系
axes[1, 0].scatter(df['AveRooms'], df['房价'], alpha=0.5, color='green')
axes[1, 0].set_xlabel('平均房间数')
axes[1, 0].set_ylabel('房价中位数 (万美元)')
axes[1, 0].set_title('房间数 vs 房价')

# 绘制人口与房价的关系
axes[1, 1].scatter(df['Population'], df['房价'], alpha=0.5, color='red')
axes[1, 1].set_xlabel('人口')
axes[1, 1].set_ylabel('房价中位数 (万美元)')
axes[1, 1].set_title('人口 vs 房价')

plt.tight_layout()
plt.show()

# 4. 准备数据用于模型训练
print("\n" + "=" * 50)
print("第三步：准备数据用于模型训练")
print("=" * 50)

# 选择特征和目标变量
# 这里我们选择收入中位数作为主要特征（单变量线性回归）
X = df[['MedInc']]  # 特征矩阵 (注意要使用双括号，使其保持二维结构)
y = df['房价']       # 目标变量

print(f"特征数据形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 将数据分为训练集和测试集
# test_size=0.2 表示20%的数据作为测试集，80%作为训练集
# random_state=42 确保每次分割结果一致，便于教学演示
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")

# 5. 创建和训练线性回归模型
print("\n" + "=" * 50)
print("第四步：创建和训练线性回归模型")
print("=" * 50)

# 创建线性回归模型实例
model = LinearRegression()

print("开始训练模型...")
# 使用训练数据拟合模型（训练模型）
model.fit(X_train, y_train)
print("模型训练完成!")

# 6. 查看模型参数
print("\n" + "=" * 50)
print("第五步：查看模型参数")
print("=" * 50)

# 获取模型的斜率和截距
slope = model.coef_[0]  # 斜率（系数）
intercept = model.intercept_  # 截距

print(f"模型方程: 房价 = {slope:.2f} × 收入中位数 + {intercept:.2f}")
print(f"斜率 (系数): {slope:.4f}")
print(f"截距: {intercept:.4f}")

# 解释模型参数的意义
print("\n模型参数解释:")
print(f"- 斜率 {slope:.2f}: 收入每增加1万美元，房价预计上涨{slope:.2f}万美元")
print(f"- 截距 {intercept:.2f}: 当收入为0时，房价的基准值（理论值）")

# 7. 使用模型进行预测
print("\n" + "=" * 50)
print("第六步：使用模型进行预测")
print("=" * 50)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 创建一个DataFrame来比较预测值和真实值
results = pd.DataFrame({
    '实际收入': X_test['MedInc'],
    '实际房价': y_test,
    '预测房价': y_pred,
    '误差': y_test - y_pred
})

print("预测结果对比 (前10个样本):")
print(results.head(10))

# 8. 评估模型性能
print("\n" + "=" * 50)
print("第七步：评估模型性能")
print("=" * 50)

# 计算均方误差 (MSE) - 越小越好
mse = mean_squared_error(y_test, y_pred)

# 计算R²分数 (决定系数) - 越接近1越好
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"R²分数: {r2:.4f}")

# 解释评估指标
print("\n模型性能解释:")
print(f"- 均方误差 {mse:.4f}: 预测值与真实值之间的平均平方差")
print(f"- R²分数 {r2:.4f}: 模型解释了房价变异的{r2*100:.2f}%")

# 9. 可视化预测结果
print("\n" + "=" * 50)
print("第八步：可视化预测结果")
print("=" * 50)

# 创建可视化图表
plt.figure(figsize=(15, 5))

# 子图1: 回归线
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, alpha=0.5, label='训练数据')
plt.scatter(X_test, y_test, alpha=0.5, color='red', label='测试数据')

# 绘制回归线
x_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 100).reshape(-1, 1)
x_range_df = pd.DataFrame(x_range, columns=['MedInc'])
y_range_pred = model.predict(x_range_df)
plt.plot(x_range, y_range_pred, color='black', linewidth=2, label='回归线')

plt.xlabel('收入中位数 (万美元)')
plt.ylabel('房价中位数 (万美元)')
plt.title('线性回归模型')
plt.legend()

# 子图2: 预测值与真实值对比
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=2)  # 理想预测线
plt.xlabel('实际房价 (万美元)')
plt.ylabel('预测房价 (万美元)')
plt.title('预测值 vs 真实值')

# 子图3: 残差图
plt.subplot(1, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('预测房价 (万美元)')
plt.ylabel('残差 (误差)')
plt.title('残差分析')

plt.tight_layout()
plt.show()

# 10. 使用模型进行新预测
print("\n" + "=" * 50)
print("第九步：使用模型进行新预测")
print("=" * 50)

# 创建一些新的收入数据用于预测
new_incomes = np.array([2, 4, 6, 8]).reshape(-1, 1)  # 收入中位数 (万美元)
new_incomes_df = pd.DataFrame(new_incomes, columns=['MedInc'])

# 使用训练好的模型进行预测
predicted_prices = model.predict(new_incomes_df)

# 显示预测结果
print("基于收入中位数的房价预测:")
for income, price in zip(new_incomes.flatten(), predicted_prices):
    print(f"收入中位数: {income}万美元 -> 预测房价: {price:.2f}万美元")

# 11. 多变量线性回归（扩展内容）
print("\n" + "=" * 50)
print("第十步：多变量线性回归（扩展）")
print("=" * 50)

# 使用多个特征来改进模型
print("现在我们将使用所有可用特征来构建更强大的模型...")

# 选择所有特征
X_multi = df.drop('房价', axis=1)  # 所有特征
y_multi = df['房价']              # 目标变量

# 分割数据
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# 创建并训练多变量线性回归模型
multi_model = LinearRegression()
multi_model.fit(X_multi_train, y_multi_train)

# 进行预测
y_multi_pred = multi_model.predict(X_multi_test)

# 评估多变量模型
mse_multi = mean_squared_error(y_multi_test, y_multi_pred)
r2_multi = r2_score(y_multi_test, y_multi_pred)

print("\n多变量线性回归模型性能:")
print(f"均方误差 (MSE): {mse_multi:.4f}")
print(f"R²分数: {r2_multi:.4f}")

# 比较单变量和多变量模型
print("\n模型比较:")
print(f"单变量模型 (仅收入) R²: {r2:.4f}")
print(f"多变量模型 (所有特征) R²: {r2_multi:.4f}")
print(f"模型改进: {((r2_multi - r2) / r2 * 100):.2f}%")

# 12. 总结
print("\n" + "=" * 50)
print("项目总结")
print("=" * 50)

print("""
本项目完成了以下内容:
1. 加载和探索了加州房价数据集
2. 可视化分析了特征与房价的关系
3. 准备了训练数据和测试数据
4. 创建并训练了单变量线性回归模型
5. 理解了模型参数的意义
6. 使用模型进行了房价预测
7. 评估了模型的性能 (MSE和R²)
8. 可视化了预测结果和残差分析
9. 尝试了多变量线性回归模型

关键知识点:
- 线性回归用于预测连续数值
- 模型通过最小化误差来学习最佳参数
- 评估指标帮助我们了解模型的预测能力
- 多变量模型通常比单变量模型更强大

接下来可以尝试:
- 尝试使用不同的特征组合
- 处理数据中的异常值
- 探索更复杂的回归模型
""")

# 保存模型（可选）
# import joblib
# joblib.dump(model, 'linear_regression_model.pkl')
# print("模型已保存为 'linear_regression_model.pkl'")
