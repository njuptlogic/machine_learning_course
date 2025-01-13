clear;
addpath(genpath('.\libsvm-3.18'));  % SVM 工具包

% ---------- 加载数据集 ----------
% 使用 Fisher Iris 数据集
load fisheriris
[~,~,labels] = unique(species);   %# labels: 1/2/3
data = meas;                      %# 数据矩阵 (150x4)

% 数据标准化
data = (data - mean(data)) ./ std(data);

% 数据集划分（训练集和测试集）
numInst = size(data, 1);
idx = randperm(numInst);          % 随机打乱数据顺序
numTrain = 100;                   % 训练样本数量
numTest = numInst - numTrain;     % 测试样本数量
trainData = data(idx(1:numTrain), :);
testData = data(idx(numTrain+1:end), :);
trainLabel = labels(idx(1:numTrain));
testLabel = labels(idx(numTrain+1:end));

% ---------- PCA降维（用于可视化） ----------
[coeff, trainDataPCA] = pca(trainData);  % 训练集 PCA
testDataPCA = testData * coeff(:, 1:2);  % 测试集降到二维

% ---------- 实验 1：SVM 分类并可视化 ----------
disp('--- SVM 分类 ---');

% 调节 SVM 参数
C = 1;    % 正则化参数
gamma = 0.5;  % RBF 核参数

% 训练 SVM 模型（One-vs-All）
model = cell(max(labels), 1);
for k = 1:max(labels)
    model{k} = svmtrain(double(trainLabel == k), trainData, sprintf('-c %f -g %f -b 1', C, gamma));
end

% 测试 SVM 模型
prob = zeros(numTest, max(labels));
for k = 1:max(labels)
    [~,~,p] = svmpredict(double(testLabel == k), testData, model{k}, '-b 1');
    prob(:,k) = p(:, model{k}.Label == 1);  % 获取概率
end

% 预测类别
[~,pred] = max(prob, [], 2);

% 输出分类准确率
acc = sum(pred == testLabel) / numel(testLabel);
disp(['SVM 准确率: ' num2str(acc)]);

% 混淆矩阵和可视化
C_matrix = confusionmat(testLabel, pred);
disp('混淆矩阵:');
disp(C_matrix);

% ---------- 可视化 1：SVM 决策边界 ----------
% 创建网格点
[x1Grid, x2Grid] = meshgrid(linspace(min(trainDataPCA(:,1)), max(trainDataPCA(:,1)), 100), ...
                            linspace(min(trainDataPCA(:,2)), max(trainDataPCA(:,2)), 100));
gridData = [x1Grid(:), x2Grid(:)];

% 计算网格点的分类结果
probGrid = zeros(size(gridData, 1), max(labels));
for k = 1:max(labels)
    [~,~,p] = svmpredict(zeros(size(gridData, 1), 1), gridData * coeff(:, 1:2)', model{k}, '-b 1');
    probGrid(:,k) = p(:, model{k}.Label == 1);  % 获取概率
end
[~,gridPred] = max(probGrid, [], 2);

% 绘制决策边界
figure;
gscatter(trainDataPCA(:,1), trainDataPCA(:,2), trainLabel, 'rgb', 'osd');
hold on;
contourf(x1Grid, x2Grid, reshape(gridPred, size(x1Grid)), 'LineColor', 'none');  % 无需 Alpha 参数
alpha(0.5);  % 调整透明度
title(sprintf('SVM with RBF Kernel (C=%.1f, gamma=%.1f)', C, gamma));
xlabel('PCA Component 1');
ylabel('PCA Component 2');
legend('Class 1', 'Class 2', 'Class 3');
hold off;

% ---------- 可视化 2：混淆矩阵 ----------
figure;
imagesc(C_matrix);
colorbar;
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
xticks(1:max(labels));
yticks(1:max(labels));
