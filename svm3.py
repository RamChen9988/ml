import os
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

def load_or_create_sample_data():
    """
    尝试加载LFW数据集，如果失败则创建模拟数据用于演示
    """
    try:
        print("正在尝试下载LFW人脸数据集...")
        faces = fetch_lfw_people(min_faces_per_person=60, download_if_missing=True)
        print(f"成功加载LFW数据集: {faces.data.shape[0]} 个样本, {faces.data.shape[1]} 个特征")
        return faces.data, faces.target, faces.target_names
    except Exception as e:
        print(f"LFW数据集下载失败: {e}")
        print("创建模拟数据用于演示...")
        
        # 创建模拟数据
        n_samples = 1000
        n_features = 2914  # LFW数据集的典型特征数
        n_classes = 7      # 模拟7个不同的人
        
        # 生成模拟数据
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # 为每个类别添加不同的模式
        for i in range(n_classes):
            class_indices = np.arange(i * 143, (i + 1) * 143)  # 大致均匀分布
            if i < n_classes - 1:
                X[class_indices] += i * 0.5  # 为每个类别添加偏移
        
        y = np.repeat(np.arange(n_classes), 143)[:n_samples]
        target_names = [f'Person_{i}' for i in range(n_classes)]
        
        print(f"创建模拟数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征, {len(target_names)} 个类别")
        return X, y, target_names

# 加载数据
X, y, target_names = load_or_create_sample_data()

# 使用PCA降维 + SVM分类的管道
pca = PCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# 分割数据并训练
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 参数调优
param_grid = {'svc__C': [1, 5, 10],
              'svc__gamma': [0.0001, 0.0005, 0.001]}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("最佳参数：", grid.best_params_)
print("测试集准确率：", grid.score(X_test, y_test))
