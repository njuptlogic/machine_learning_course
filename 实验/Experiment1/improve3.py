import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

# 导入手写数字数据集
digits = datasets.load_digits()
X = digits.data  # 高维特征 (64维)
Y = digits.target  # 标签 (0-9 共10类)

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 使用 RBF 核的 SVM 模型
C = 100  # 正则化参数
gamma = 0.001  # RBF 核的参数
rbf_svc = SVC(kernel='rbf', C=C, gamma=gamma)

# 训练模型
rbf_svc.fit(X_train, Y_train)

# 测试模型
Y_pred = rbf_svc.predict(X_test)

# 输出分类结果
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
print(f"Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")

# 可视化：使用 PCA 将高维数据降到二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
rbf_svc_pca = SVC(kernel='rbf', C=C, gamma=gamma)  # 再训练一个用于二维数据的模型
rbf_svc_pca.fit(X_train_pca, Y_train)

# 创建网格点，用于绘制决策边界
h = .02  # 网格的步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 预测网格点的值，用于决策边界的绘制
Z = rbf_svc_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
pl.figure(figsize=(8, 6))
pl.contourf(xx, yy, Z, alpha=0.8, cmap=pl.cm.Paired)
pl.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, edgecolors='k', marker='o', s=50, cmap=pl.cm.Paired)
pl.title(f"SVM with RBF Kernel - Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")
pl.xlabel("PCA Component 1")
pl.ylabel("PCA Component 2")
pl.show()
