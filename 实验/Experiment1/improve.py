import numpy as np
import pylab as pl
from sklearn import svm, datasets
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# 导入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 选择前两个特征
Y = iris.target

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 定义 SVM 和其他相关算法
C = 20  # SVM 正则化参数
gamma = 1.8  # 核函数的参数   0.7
degree = 2  # 多项式核函数的度数   3

svc = svm.SVC(kernel='linear', C=C)
rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C)
poly_svc = svm.SVC(kernel='poly', degree=degree, C=C)
lin_svc = svm.LinearSVC(C=C)

# 用其他分类算法进行对比（例如：K近邻、决策树）
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

knn = KNeighborsClassifier(n_neighbors=3)
dt = DecisionTreeClassifier(random_state=42)

# 在不同模型上训练
models = [svc, rbf_svc, poly_svc, lin_svc, knn, dt]
model_names = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial kernel', 'LinearSVC', 'KNN',
               'Decision Tree']

# 创建一个网格来绘制决策边界
h = .02  # 网格的步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 输出每个模型的评价结果
for model, name in zip(models, model_names):
    model.fit(X_train, Y_train)  # 训练模型
    Y_pred = model.predict(X_test)  # 预测结果

    # 打印分类报告
    print(f"Model: {name}")
    print(classification_report(Y_test, Y_pred))

    # 画决策边界图
    pl.subplot(2, 3, models.index(model) + 1)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, alpha=0.8)
    pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o', s=50)
    pl.title(f"{name} - Accuracy: {accuracy_score(Y_test, Y_pred):.2f}")

pl.tight_layout()
pl.show()
