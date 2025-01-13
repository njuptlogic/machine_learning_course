import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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

# 存储评价结果
evaluation_results = []

# 遍历每种聚类模型
for name, est in estimators:
    # 拟合聚类模型
    est.fit(X)
    labels = est.labels_

    # 计算聚类评价指标
    silhouette_avg = silhouette_score(X, labels)  # 轮廓系数
    ch_score = calinski_harabasz_score(X, labels)  # Calinski-Harabasz 指数
    db_score = davies_bouldin_score(X, labels)  # Davies-Bouldin 指数

    # 存储结果
    evaluation_results.append((name, silhouette_avg, ch_score, db_score))

    # 打印聚类评价结果
    print(f"Clustering evaluation for {name}:")
    print(f"  Silhouette Score: {silhouette_avg:.3f}")
    print(f"  Calinski-Harabasz Index: {ch_score:.3f}")
    print(f"  Davies-Bouldin Index: {db_score:.3f}")
    print("-" * 50)

# 绘制聚类结果的评价标准
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
metrics = ["Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"]

for i, metric in enumerate(metrics):
    ax[i].bar(
        [result[0] for result in evaluation_results],
        [result[i + 1] for result in evaluation_results],
        color=["blue", "green", "orange"],
    )
    ax[i].set_title(metric)
    ax[i].set_xticklabels([result[0] for result in evaluation_results], rotation=15)
    ax[i].set_ylabel("Score")

plt.tight_layout()
plt.show()
