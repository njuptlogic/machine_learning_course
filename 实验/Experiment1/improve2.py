import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 导入手写数字数据集
digits = datasets.load_digits()
X = digits.data
Y = digits.target

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 定义 SVM 和其他相关算法
C = 100  # SVM 正则化参数
gamma = 0.7  # 核函数的参数
degree = 3  # 多项式核函数的度数

svc = svm.SVC(kernel='linear', C=C)
rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C)
poly_svc = svm.SVC(kernel='poly', degree=degree, C=C)
lin_svc = svm.LinearSVC(C=C)

# 使用PCA降维，便于绘制决策边界图
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 模型训练
models = [svc, rbf_svc, poly_svc, lin_svc]
model_names = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial kernel', 'LinearSVC']

# 训练每个模型并绘制决策边界
pl.figure(figsize=(12, 12))

for i, model in enumerate(models):
    # 训练模型
    model.fit(X_train_pca, Y_train)

    # 预测结果
    Y_pred = model.predict(X_test_pca)

    # 打印分类报告
    print(f"Model: {model_names[i]}")
    print(classification_report(Y_test, Y_pred))

    # 创建网格点，便于绘制决策边界
    h = .02  # 网格的步长
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 绘制决策边界
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 子图
    pl.subplot(2, 2, i + 1)
    pl.contourf(xx, yy, Z, alpha=0.8)
    pl.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=Y_train, edgecolors='k', marker='o', s=50)
    pl.title(f"{model_names[i]} - Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")

# 显示图形
pl.tight_layout()
pl.show()
