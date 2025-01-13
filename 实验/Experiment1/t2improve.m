clear;
load fisheriris
X = meas(:,3:4); % 选择花瓣长度和宽度两列作为特征

% 原始数据可视化
figure;
plot(X(:,1), X(:,2), 'k*', 'MarkerSize', 5);
title 'Fisher''s Iris Data';
xlabel 'Petal Lengths (cm)';
ylabel 'Petal Widths (cm)';

% ---------- 参数设置 ----------
rng(1); % 设置随机数种子，保证可复现性
k_values = [2, 3, 4]; % 尝试不同的聚类数目 k
distance_metrics = {'sqeuclidean', 'cityblock', 'cosine'}; % 距离度量方式
max_iterations = [50, 100, 200]; % 最大迭代次数

% ---------- 不同参数组合的实验 ----------
figure;
plot_num = 1; % 用于子图编号

for k = k_values
    for dist = distance_metrics
        for max_iter = max_iterations
            % 执行 k-means 聚类
            [idx, C] = kmeans(X, k, 'Distance', dist{1}, 'MaxIter', max_iter);

            % 创建网格，用于决策区域的可视化
            x1 = min(X(:,1)):0.01:max(X(:,1));
            x2 = min(X(:,2)):0.01:max(X(:,2));
            [x1G, x2G] = meshgrid(x1, x2);
            XGrid = [x1G(:), x2G(:)]; % 定义一个密集网格

            % 计算决策区域
            idx2Region = kmeans(XGrid, k, 'MaxIter', 1, 'Start', C);

            % 可视化聚类结果
            subplot(length(k_values), length(distance_metrics), plot_num);
            gscatter(XGrid(:,1), XGrid(:,2), idx2Region, ...
                parula(k), '..'); % 使用 colormap 可视化区域
            hold on;
            plot(X(:,1), X(:,2), 'k*', 'MarkerSize', 5); % 数据点
            plot(C(:,1), C(:,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); % 聚类中心
            title(sprintf('k=%d, Dist=%s, MaxIter=%d', k, dist{1}, max_iter));
            xlabel 'Petal Lengths (cm)';
            ylabel 'Petal Widths (cm)';
            legend off;
            hold off;
            
            % 更新子图编号
            plot_num = plot_num + 1;
        end
    end
end

% 调整图表布局
sgtitle('K-means Clustering with Different Parameters');
