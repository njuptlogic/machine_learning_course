import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

# 加载 Iris 数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 定义不同的 KMeans 聚类器
estimators = [
    ('k_means_iris_8', KMeans(n_clusters=8)),
    ('k_means_iris_3', KMeans(n_clusters=3)),
    ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1, init='random'))
]

labels_list = []
fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

# 遍历每种聚类模型并可视化
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
    fig.add_axes(ax)

    # 拟合聚类模型
    est.fit(X)
    labels = est.labels_
    labels_list.append(labels)

    # 绘制 3D 数据点（根据聚类结果上色）
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolors='k')

    # 设置坐标轴与标题
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12

    fignum += 1
    plt.show()

# 绘制真实分类标签的 3D 图
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
fig.add_axes(ax)

# 添加类别名称
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# 调整标签顺序，使颜色匹配
y = np.choose(y, [1, 2, 0]).astype(float)  # 替换 np.float 为 float
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolors='k')

# 设置坐标轴与标题
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

plt.show()
