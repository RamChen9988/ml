# -*- coding: utf-8 -*-
"""
员工离职预测实战项目 - 逻辑回归在企业人力资源分析中的应用
适用对象：有一定基础的学生，模拟真实工作场景
作者：AI教师
日期：2024年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve,
                            precision_recall_curve, average_precision_score)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("员工离职预测实战项目 - 企业人力资源分析")
print("=" * 70)

# 1. 数据加载和业务理解
print("\n第一步：数据加载和业务理解")
print("-" * 50)

# 创建模拟的员工数据集（在实际工作中，这会从公司数据库获取）
def create_employee_data(n_samples=1500):
    """创建模拟的员工数据集"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(22, 60, n_samples),
        'department': np.random.choice(['销售', '技术', '市场', '人力资源', '财务', '运营'], n_samples),
        'salary_level': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'work_years': np.random.exponential(5, n_samples).astype(int) + 1,
        'overtime_hours': np.random.poisson(10, n_samples),
        'satisfaction_score': np.random.normal(6.5, 2, n_samples).clip(1, 10),
        'last_performance_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.3, 0.3, 0.2]),
        'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'training_hours': np.random.poisson(25, n_samples),
        'work_accident': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # 基于业务规则创建离职标签（在实际工作中这是真实的历史数据）
    # 离职概率受到多个因素影响
    resignation_prob = (
        0.1 + 
        0.02 * (df['age'] < 30) + 
        0.05 * (df['salary_level'] == 1) +
        0.03 * (df['overtime_hours'] > 15) +
        0.08 * (df['satisfaction_score'] < 4) +
        0.06 * (df['last_performance_rating'] == 1) -
        0.04 * (df['promotion_last_5years'] == 1) -
        0.02 * (df['training_hours'] > 30)
    )
    
    df['resigned'] = np.random.binomial(1, resignation_prob.clip(0, 0.8))
    
    return df

# 创建数据集
print("正在生成员工数据集...")
df = create_employee_data()
print(f"数据集形状: {df.shape}")
print(f"离职率: {df['resigned'].mean():.2%}")

print("\n数据字段说明:")
print("- age: 年龄")
print("- department: 部门")
print("- salary_level: 薪资等级 (1-5)")
print("- work_years: 工作年限")
print("- overtime_hours: 月均加班小时")
print("- satisfaction_score: 工作满意度 (1-10)")
print("- last_performance_rating: 最近绩效评级 (1-5)")
print("- promotion_last_5years: 近5年是否晋升")
print("- training_hours: 年培训小时数")
print("- work_accident: 是否发生过工作事故")
print("- resigned: 是否离职 (目标变量)")

# 2. 探索性数据分析
print("\n第二步：探索性数据分析")
print("-" * 50)

print("\n数据基本信息:")
print(df.info())

print("\n数值型变量描述统计:")
print(df.describe())

print("\n离职情况统计:")
resignation_stats = df['resigned'].value_counts()
print(resignation_stats)
print(f"离职率: {resignation_stats[1] / len(df):.2%}")

# 离职率分析可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('员工离职因素分析', fontsize=16)

# 各部门离职率
dept_resign = df.groupby('department')['resigned'].mean().sort_values(ascending=False)
axes[0, 0].bar(dept_resign.index, dept_resign.values, color='skyblue')
axes[0, 0].set_title('各部门离职率')
axes[0, 0].tick_params(axis='x', rotation=45)

# 薪资等级与离职率
salary_resign = df.groupby('salary_level')['resigned'].mean()
axes[0, 1].plot(salary_resign.index, salary_resign.values, marker='o', color='orange')
axes[0, 1].set_title('薪资等级 vs 离职率')
axes[0, 1].set_xlabel('薪资等级')
axes[0, 1].set_ylabel('离职率')

# 工作年限与离职率
work_years_resign = df.groupby('work_years')['resigned'].mean()
axes[0, 2].plot(work_years_resign.index, work_years_resign.values, marker='s', color='green')
axes[0, 2].set_title('工作年限 vs 离职率')
axes[0, 2].set_xlabel('工作年限')

# 满意度分布
df[df['resigned'] == 0]['satisfaction_score'].hist(alpha=0.7, label='在职', ax=axes[1, 0], bins=20)
df[df['resigned'] == 1]['satisfaction_score'].hist(alpha=0.7, label='离职', ax=axes[1, 0], bins=20)
axes[1, 0].set_title('工作满意度分布')
axes[1, 0].legend()

# 绩效评级与离职率
performance_resign = df.groupby('last_performance_rating')['resigned'].mean()
axes[1, 1].bar(performance_resign.index, performance_resign.values, color='purple')
axes[1, 1].set_title('绩效评级 vs 离职率')
axes[1, 1].set_xlabel('绩效评级')

# 加班时长分布
df[df['resigned'] == 0]['overtime_hours'].hist(alpha=0.7, label='在职', ax=axes[1, 2], bins=15)
df[df['resigned'] == 1]['overtime_hours'].hist(alpha=0.7, label='离职', ax=axes[1, 2], bins=15)
axes[1, 2].set_title('加班时长分布')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# 3. 特征工程
print("\n第三步：特征工程")
print("-" * 50)

# 复制原始数据
df_processed = df.copy()

# 对分类变量进行编码
print("对分类变量进行编码...")
label_encoders = {}
categorical_columns = ['department']

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"{col} 编码完成: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 创建新特征（基于业务理解）
print("\n创建新业务特征...")
df_processed['tenure_group'] = pd.cut(df_processed['work_years'], 
                                     bins=[0, 2, 5, 10, 50], 
                                     labels=['新人(0-2年)', '成长(2-5年)', '稳定(5-10年)', '资深(10+年)'])
df_processed['satisfaction_group'] = pd.cut(df_processed['satisfaction_score'],
                                           bins=[0, 4, 7, 10],
                                           labels=['低满意度', '中等满意度', '高满意度'])
df_processed['overtime_intensity'] = df_processed['overtime_hours'] / df_processed['work_years'].clip(1)

# 对新创建的分类特征进行编码
new_categorical = ['tenure_group', 'satisfaction_group']
for col in new_categorical:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print("特征工程完成!")
print(f"处理后的数据形状: {df_processed.shape}")

# 4. 特征选择和数据集准备
print("\n第四步：特征选择和数据集准备")
print("-" * 50)

# 选择特征（排除目标变量和原始分类变量）
feature_columns = [col for col in df_processed.columns if col not in 
                  ['resigned', 'tenure_group', 'satisfaction_group']]

X = df_processed[feature_columns]
y = df_processed['resigned']

print(f"特征数量: {len(feature_columns)}")
print("使用的特征:", feature_columns)

# 使用ANOVA F-value进行特征选择
print("\n进行特征选择...")
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print(f"选择的前{len(selected_features)}个重要特征:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# 使用选择的特征
X = df_processed[selected_features]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")
print(f"训练集离职率: {y_train.mean():.2%}")
print(f"测试集离职率: {y_test.mean():.2%}")

# 5. 数据标准化
print("\n第五步：数据标准化")
print("-" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("数据标准化完成!")

# 6. 模型训练和超参数调优
print("\n第六步：模型训练和超参数调优")
print("-" * 50)

# 定义参数网格
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

print("开始网格搜索寻找最佳参数...")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数 (AUC):", grid_search.best_score_)

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

print("最终模型训练完成!")

# 7. 模型评估
print("\n第七步：模型评估")
print("-" * 50)

# 预测
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

print(f"准确率: {accuracy:.4f}")
print(f"AUC得分: {auc_score:.4f}")
print(f"平均精确率: {average_precision:.4f}")

print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['在职', '离职']))

# 8. 业务解读和可视化
print("\n第八步：业务解读和可视化")
print("-" * 50)

# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': np.abs(best_model.coef_[0])
}).sort_values('importance', ascending=False)

print("\n特征重要性排序:")
print(feature_importance)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 特征重要性
axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
axes[0, 0].set_title('特征重要性')
axes[0, 0].set_xlabel('重要性（系数绝对值）')

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['预测在职', '预测离职'],
            yticklabels=['实际在职', '实际离职'])
axes[0, 1].set_title('混淆矩阵')

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('假正率')
axes[1, 0].set_ylabel('真正率')
axes[1, 0].set_title('ROC曲线')
axes[1, 0].legend(loc="lower right")

# 精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1, 1].plot(recall, precision, color='green', lw=2, 
                label=f'PR曲线 (AP = {average_precision:.3f})')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('召回率')
axes[1, 1].set_ylabel('精确率')
axes[1, 1].set_title('精确率-召回率曲线')
axes[1, 1].legend(loc="upper right")

plt.tight_layout()
plt.show()

# 9. 业务应用：高风险员工识别
print("\n第九步：高风险员工识别")
print("-" * 50)

# 对整个数据集进行预测
all_predictions = best_model.predict_proba(scaler.transform(X))[:, 1]
df['resignation_risk'] = all_predictions

# 识别高风险员工（预测概率 > 0.7）
high_risk_employees = df[df['resignation_risk'] > 0.7].copy()
high_risk_employees = high_risk_employees.sort_values('resignation_risk', ascending=False)

print(f"识别到 {len(high_risk_employees)} 名高风险员工 (预测离职概率 > 70%)")
print("\n高风险员工特征分析:")

if len(high_risk_employees) > 0:
    print(f"- 平均满意度: {high_risk_employees['satisfaction_score'].mean():.2f}")
    print(f"- 平均加班时长: {high_risk_employees['overtime_hours'].mean():.2f} 小时")
    print(f"- 平均工作年限: {high_risk_employees['work_years'].mean():.2f} 年")
    print(f"- 薪资等级分布: {high_risk_employees['salary_level'].value_counts().to_dict()}")
    
    print("\n前10名最高风险员工:")
    display_cols = ['age', 'department', 'salary_level', 'work_years', 
                   'satisfaction_score', 'resignation_risk']
    print(high_risk_employees[display_cols].head(10).round(3))
else:
    print("未发现高风险员工")

# 10. 模型部署和应用建议
print("\n第十步：模型部署和应用建议")
print("-" * 50)

print("""
业务应用建议:

1. 预警系统
   - 将模型集成到HR系统中，定期评估员工离职风险
   - 对高风险员工进行标记和重点关注

2. 针对性干预
   - 对高风险员工进行一对一沟通
   - 分析具体原因（满意度低、加班多、薪资低等）
   - 制定个性化保留方案

3. 组织改进
   - 分析高风险部门的共同问题
   - 改进管理制度和工作环境
   - 优化薪资和晋升体系

4. 持续优化
   - 定期重新训练模型
   - 收集反馈数据改进预测准确性
   - 跟踪干预措施的效果
""")

# 11. 保存模型和预处理工具（在实际工作中使用）
print("\n第十一步：模型保存")
print("-" * 50)

# 在实际部署中，我们会保存以下内容:
# - 训练好的模型
# - 标准化器
# - 特征选择器  
# - 标签编码器

print("""
在实际部署中需要保存:
✓ 训练好的逻辑回归模型
✓ 标准化器 (StandardScaler)
✓ 特征选择器 (SelectKBest)
✓ 标签编码器 (LabelEncoder)

保存代码示例:
import joblib
joblib.dump(best_model, 'employee_attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
""")

# 12. 项目总结
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print(f"""
项目完成情况:

数据准备:
  - 员工记录: {len(df)} 条
  - 特征数量: {len(selected_features)} 个
  - 整体离职率: {df['resigned'].mean():.2%}

模型性能:
  - 准确率: {accuracy:.4f}
  - AUC得分: {auc_score:.4f}
  - 平均精确率: {average_precision:.4f}

业务成果:
  - 识别高风险员工: {len(high_risk_employees)} 名
  - 关键影响因素: {', '.join(feature_importance['feature'].head(3).tolist())}

后续行动:
  1. 验证模型在真实环境中的表现
  2. 与HR部门合作制定干预策略
  3. 建立定期评估和模型更新机制
  4. 扩展更多特征提高预测准确性
""")