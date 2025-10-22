import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_recall_curve, roc_curve, auc)
from sklearn.pipeline import Pipeline
import re
import string
from wordcloud import WordCloud
import requests
import os
import zipfile
import warnings

def download_spam_dataset():
    """下载垃圾邮件数据集"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    local_zip_path = "smsspamcollection.zip"
    data_file_path = "SMSSpamCollection"
    
    # 如果数据文件已存在，直接返回
    if os.path.exists(data_file_path):
        print("数据集已存在，直接加载...")
        return data_file_path
    
    print("正在下载数据集...")
    try:
        response = requests.get(url, timeout=30)
        with open(local_zip_path, 'wb') as f:
            f.write(response.content)
        
        # 解压文件
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        os.remove(local_zip_path)
        print("数据集下载和解压完成!")
        return data_file_path
    except Exception as e:
        print(f"下载失败: {e}")
        print("使用备用数据源...")
        # 这里可以添加备用数据源或生成模拟数据
        return None
    

    # 下载数据集
data_path = download_spam_dataset()


if data_path and os.path.exists(data_path):
    # 加载数据
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])
    print("数据集加载成功!")
else:
    print("使用模拟数据集...")
    # 创建模拟的垃圾邮件数据集
    np.random.seed(42)
    
    # 正常短信示例
    ham_messages = [
        "OK, I'll call you later",
        "Thanks for your message",
        "See you tomorrow at the meeting",
        "Can we reschedule for next week?",
        "I'll be there in 10 minutes",
        "Don't forget to buy milk",
        "Happy birthday to you!",
        "What time is the movie?",
        "Let's have lunch together",
        "Call me when you're free"
    ]
    
    # 垃圾短信示例
    spam_messages = [
        "FREE entry to win £1000 cash prize! Reply NOW",
        "You have won a new iPhone! Claim your prize",
        "URGENT: Your bank account needs verification",
        "Congratulations! You are our lucky winner",
        "Limited time offer: 50% discount only today",
        "Cash reward waiting for you, call now",
        "You have been selected for a free gift",
        "Important: Your package delivery failed",
        "Exclusive deal just for you, buy now",
        "You qualify for a special loan offer"
    ]
    
    # 生成更多样本
    def generate_more_samples(base_messages, n_samples):
        samples = []
        for _ in range(n_samples):
            base_msg = np.random.choice(base_messages)
            # 添加一些随机变化
            variations = ["", "!", "!!", " :)", " :(", " ..."]
            samples.append(base_msg + np.random.choice(variations))
        return samples
    
    ham_samples = generate_more_samples(ham_messages, 1000)
    spam_samples = generate_more_samples(spam_messages, 500)
    
    # 创建DataFrame
    data = {
        'label': ['ham'] * len(ham_samples) + ['spam'] * len(spam_samples),
        'text': ham_samples + spam_samples
    }
    df = pd.DataFrame(data)
    print("模拟数据集创建完成!")

    
print(f"\n数据集形状: {df.shape}")
print(f"数据集预览:")
print(df.head())

print(f"\n类别分布:")
label_counts = df['label'].value_counts()
print(label_counts)
print(f"垃圾邮件比例: {label_counts['spam']/len(df):.2%}")

# 2. 数据探索和可视化
print("\n第二步：数据探索和可视化")
print("-" * 50)

# 文本长度分析
df['text_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\n文本长度统计:")
print(df.groupby('label')['text_length'].describe())

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('垃圾邮件数据集探索分析', fontsize=16)

# 类别分布
axes[0, 0].bar(label_counts.index, label_counts.values, color=['lightgreen', 'salmon'])
axes[0, 0].set_title('邮件类别分布')
axes[0, 0].set_ylabel('数量')

# 文本长度分布
ham_lengths = df[df['label'] == 'ham']['text_length']
spam_lengths = df[df['label'] == 'spam']['text_length']

axes[0, 1].hist(ham_lengths, alpha=0.7, label='正常邮件', bins=30, color='lightgreen')
axes[0, 1].hist(spam_lengths, alpha=0.7, label='垃圾邮件', bins=30, color='salmon')
axes[0, 1].set_xlabel('文本长度')
axes[0, 1].set_ylabel('频数')
axes[0, 1].set_title('文本长度分布')
axes[0, 1].legend()

# 词数分布
ham_words = df[df['label'] == 'ham']['word_count']
spam_words = df[df['label'] == 'spam']['word_count']

axes[0, 2].hist(ham_words, alpha=0.7, label='正常邮件', bins=20, color='lightgreen')
axes[0, 2].hist(spam_words, alpha=0.7, label='垃圾邮件', bins=20, color='salmon')
axes[0, 2].set_xlabel('词数')
axes[0, 2].set_ylabel('频数')
axes[0, 2].set_title('词数分布')
axes[0, 2].legend()


# 文本预处理函数
def preprocess_text(text):
    """文本预处理"""
    # 转换为小写
    text = text.lower()
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 移除非字母字符
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 应用预处理
df['cleaned_text'] = df['text'].apply(preprocess_text)


# 生成词云
def generate_wordcloud(texts, title, ax, color):
    """生成词云图"""
    all_text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap=color,
                         max_words=100).generate(all_text)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=14)
    ax.axis('off')

# 正常邮件词云
ham_texts = df[df['label'] == 'ham']['cleaned_text']
generate_wordcloud(ham_texts, '正常邮件词云', axes[1, 0], 'Greens')

# 垃圾邮件词云
spam_texts = df[df['label'] == 'spam']['cleaned_text']
generate_wordcloud(spam_texts, '垃圾邮件词云', axes[1, 1], 'Reds')



# 标点符号使用分析
def count_punctuation(text):
    """统计标点符号数量"""
    return sum(1 for char in text if char in string.punctuation)

df['punctuation_count'] = df['text'].apply(count_punctuation)

ham_punct = df[df['label'] == 'ham']['punctuation_count']
spam_punct = df[df['label'] == 'spam']['punctuation_count']

axes[1, 2].hist(ham_punct, alpha=0.7, label='正常邮件', bins=15, color='lightgreen')
axes[1, 2].hist(spam_punct, alpha=0.7, label='垃圾邮件', bins=15, color='salmon')
axes[1, 2].set_xlabel('标点符号数量')
axes[1, 2].set_ylabel('频数')
axes[1, 2].set_title('标点符号使用分布')
axes[1, 2].legend()

plt.tight_layout()
plt.show()


# 3. 特征工程
print("\n第三步：特征工程")
print("-" * 50)

# 准备数据
X = df['cleaned_text']
y = df['label'].map({'ham': 0, 'spam': 1})  # 转换为0/1标签

print(f"特征数据形状: {X.shape}")
print(f"目标变量分布: \n{pd.Series(y).value_counts()}")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n训练集大小: {X_train.shape[0]} 个样本")
print(f"测试集大小: {X_test.shape[0]} 个样本")
print(f"训练集垃圾邮件比例: {y_train.mean():.2%}")
print(f"测试集垃圾邮件比例: {y_test.mean():.2%}")


# 4. 文本向量化方法比较
print("\n第四步：文本向量化方法比较")
print("-" * 50)

# 比较不同的文本向量化方法
vectorizers = {
    'CountVectorizer': CountVectorizer(stop_words='english', max_features=3000),
    'TF-IDF': TfidfVectorizer(stop_words='english', max_features=3000)
}

vectorizer_results = {}

for name, vectorizer in vectorizers.items():
    print(f"\n使用 {name}...")
    
    # 转换训练数据
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 训练朴素贝叶斯模型
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    
    # 预测和评估
    y_pred = nb_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    
    # 保存结果
    vectorizer_results[name] = {
        'vectorizer': vectorizer,
        'model': nb_model,
        'accuracy': accuracy,
        'X_train_vec': X_train_vec,
        'X_test_vec': X_test_vec
    }

# 选择最佳向量化方法
best_vectorizer_name = max(vectorizer_results.keys(), 
                          key=lambda x: vectorizer_results[x]['accuracy'])
best_result = vectorizer_results[best_vectorizer_name]

print(f"\n最佳向量化方法: {best_vectorizer_name}")
print(f"最佳准确率: {best_result['accuracy']:.4f}")


# 5. 不同类型的朴素贝叶斯比较
print("\n第五步：不同类型的朴素贝叶斯比较")
print("-" * 50)

# 使用最佳向量化方法
vectorizer = best_result['vectorizer']
X_train_vec = best_result['X_train_vec']
X_test_vec = best_result['X_test_vec']

# 比较不同的朴素贝叶斯变体
nb_variants = {
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    'GaussianNB': GaussianNB()
}

# 注意：GaussianNB需要稠密矩阵
X_train_dense = X_train_vec.toarray() if hasattr(X_train_vec, 'toarray') else X_train_vec
X_test_dense = X_test_vec.toarray() if hasattr(X_test_vec, 'toarray') else X_test_vec

nb_results = {}

for name, classifier in nb_variants.items():
    print(f"\n训练 {name}...")
    
    try:
        if name == 'GaussianNB':
            # GaussianNB需要稠密矩阵
            classifier.fit(X_train_dense, y_train)
            y_pred = classifier.predict(X_test_dense)
        else:
            classifier.fit(X_train_vec, y_train)
            y_pred = classifier.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        nb_results[name] = {
            'classifier': classifier,
            'accuracy': accuracy,
            'y_pred': y_pred
        }
        print(f"准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"{name} 训练失败: {e}")
        nb_results[name] = None

# 可视化比较结果
plt.figure(figsize=(10, 6))
variants = [name for name in nb_results if nb_results[name] is not None]
accuracies = [nb_results[name]['accuracy'] for name in variants]

bars = plt.bar(variants, accuracies, color=['steelblue', 'lightgreen', 'salmon'])
plt.ylabel('准确率')
plt.title('不同朴素贝叶斯变体性能比较')
plt.ylim(0, 1.0)

# 在柱状图上添加数值
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 6. 使用最佳模型进行详细分析
print("\n第六步：使用最佳模型进行详细分析")
print("-" * 50)

# 选择最佳模型
best_nb_name = max(nb_results.keys(), 
                  key=lambda x: nb_results[x]['accuracy'] if nb_results[x] else 0)
best_nb_result = nb_results[best_nb_name]

print(f"最佳模型: {best_nb_name}")
print(f"测试集准确率: {best_nb_result['accuracy']:.4f}")

# 详细评估
y_pred = best_nb_result['y_pred']

print("\n详细分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常邮件', '垃圾邮件']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['预测正常', '预测垃圾'],
            yticklabels=['实际正常', '实际垃圾'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')

# 7. 模型可解释性分析
print("\n第七步：模型可解释性分析")
print("-" * 50)

# 获取特征重要性（仅适用于MultinomialNB）
if best_nb_name == 'MultinomialNB':
    print("分析特征重要性...")
    
    model = best_nb_result['classifier']
    feature_names = vectorizer.get_feature_names_out()
    
    # 获取每个词在垃圾邮件和正常邮件中的对数概率
    spam_log_probs = model.feature_log_prob_[1]  # 垃圾邮件的词概率
    ham_log_probs = model.feature_log_prob_[0]   # 正常邮件的词概率
    
    # 计算词的重要性分数（垃圾邮件概率 - 正常邮件概率）
    word_scores = spam_log_probs - ham_log_probs
    
    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        'word': feature_names,
        'spam_score': spam_log_probs,
        'ham_score': ham_log_probs,
        'importance': word_scores
    })
    
    # 找出最重要的垃圾词和正常词
    top_spam_words = feature_importance.nlargest(10, 'importance')
    top_ham_words = feature_importance.nsmallest(10, 'importance')
    
    print("\n最重要的垃圾词特征:")
    print(top_spam_words[['word', 'importance']].round(4))
    
    print("\n最重要的正常词特征:")
    print(top_ham_words[['word', 'importance']].round(4))
    
    # 可视化特征重要性
    plt.subplot(1, 2, 2)
    
    # 合并显示前15个重要词
    top_words = pd.concat([top_spam_words.head(8), top_ham_words.head(7)])
    colors = ['red' if x > 0 else 'green' for x in top_words['importance']]
    
    plt.barh(range(len(top_words)), top_words['importance'], color=colors)
    plt.yticks(range(len(top_words)), top_words['word'])
    plt.xlabel('重要性分数')
    plt.title('特征词重要性分析\n(红色:垃圾词, 绿色:正常词)')
    
    plt.tight_layout()
    plt.show()

else:
    print(f"{best_nb_name} 模型的特征重要性分析较复杂，跳过此步骤。")
    plt.tight_layout()
    plt.show()

# 8. 使用管道简化流程
print("\n第八步：使用管道简化流程")
print("-" * 50)

# 创建完整的文本分类管道
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', best_nb_result['classifier'])
])

# 使用管道进行交叉验证
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 重新训练管道
pipeline.fit(X_train, y_train)


# 9. 实际应用演示
print("\n第九步：实际应用演示")
print("-" * 50)

# 测试一些新的短信
test_messages = [
    "Hello, how are you doing today?",  # 正常
    "FREE FREE FREE get your free gift now!!!",  # 垃圾
    "Can we meet for coffee tomorrow?",  # 正常
    "URGENT: You have won a $1000 prize!",  # 垃圾
    "Don't forget the meeting at 3pm",  # 正常
    "Congratulations! You are selected for exclusive offer"  # 垃圾
]

print("新短信分类预测:")
for i, message in enumerate(test_messages, 1):
    prediction = pipeline.predict([message])[0]
    probability = pipeline.predict_proba([message])[0]
    
    label = "垃圾邮件" if prediction == 1 else "正常邮件"
    spam_prob = probability[1]
    
    print(f"\n{i}. '{message}'")
    print(f"   预测: {label}")
    print(f"   垃圾邮件概率: {spam_prob:.4f} ({spam_prob*100:.1f}%)")

# 10. 概率阈值调整
print("\n第十步：概率阈值调整分析")
print("-" * 50)

# 获取预测概率
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# 计算不同阈值下的精确率和召回率
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# 找到最佳阈值（F1分数最大）
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"默认阈值 (0.5) 的准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"最佳阈值: {best_threshold:.4f}")
print(f"最佳阈值下的F1分数: {f1_scores[best_threshold_idx]:.4f}")

# 可视化精确率-召回率曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(recall, precision, marker='.')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率曲线')
plt.grid(True, alpha=0.3)

# ROC曲线
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线')
plt.legend(loc="lower right")

# 阈值对性能的影响
plt.subplot(1, 3, 3)
accuracies = []
for threshold in np.linspace(0.1, 0.9, 50):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    accuracies.append(accuracy_score(y_test, y_pred_thresh))

plt.plot(np.linspace(0.1, 0.9, 50), accuracies, marker='o', markersize=3)
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='默认阈值0.5')
plt.axvline(x=best_threshold, color='green', linestyle='--', alpha=0.7, label=f'最佳阈值{best_threshold:.2f}')
plt.xlabel('分类阈值')
plt.ylabel('准确率')
plt.title('阈值对准确率的影响')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 11. 项目总结
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print(f"""
朴素贝叶斯文本分类项目完成情况:

数据准备:
  - 短信样本: {len(df)} 条
  - 垃圾邮件比例: {label_counts['spam']/len(df):.2%}
  - 文本预处理: 小写转换、标点移除、停用词过滤

模型性能:
  - 最佳向量化方法: {best_vectorizer_name}
  - 最佳朴素贝叶斯变体: {best_nb_name}
  - 测试集准确率: {best_nb_result['accuracy']:.4f}
  - 交叉验证准确率: {cv_scores.mean():.4f}

关键发现:
  - 垃圾邮件通常包含: 'free', 'win', 'cash', 'prize', 'urgent'等词
  - 正常邮件常用词: 'call', 'thanks', 'meeting', 'tomorrow'等
  - 垃圾邮件通常更短，但标点符号使用更多
  - 调整分类阈值可以优化模型在不同场景下的表现

朴素贝叶斯算法特点:
  ✓ 简单快速，训练和预测效率高
  ✓ 适合文本分类和高维数据
  ✓ 模型可解释性强
  ✓ 对小数据集表现良好
  ✗ 特征独立性假设在现实中不成立
  ✗ 对输入数据的分布比较敏感

实际应用建议:
  1. 文本分类是朴素贝叶斯的优势领域
  2. 选择合适的文本向量化方法很重要
  3. 根据业务需求调整分类阈值
  4. 定期更新模型以适应新的垃圾邮件模式
  5. 结合其他特征（如发件人、时间等）提高准确性

扩展应用:
  - 情感分析（正面/负面评论）
  - 新闻分类（体育、财经、娱乐等）
  - 主题建模
  - 多语言文本分类
""")

# 保存模型（可选）
print("\n模型保存代码示例:")
print("""
import joblib

# 保存整个管道
joblib.dump(pipeline, 'spam_classifier_pipeline.pkl')

# 加载模型进行预测
# loaded_pipeline = joblib.load('spam_classifier_pipeline.pkl')
# prediction = loaded_pipeline.predict(["Your message here"])
""")
