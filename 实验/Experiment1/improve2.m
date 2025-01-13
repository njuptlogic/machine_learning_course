clear;
addpath(genpath('.\libsvm-3.18'));

% ---------- 加载数据集 ----------
load fisheriris
[~,~,labels] = unique(species);   %# labels: 1/2/3
data = meas;                      %# 特征数据矩阵 (150x4)
numInst = size(data,1);           % 数据样本数量
numLabels = max(labels);          % 类别数量

% ---------- 划分训练集与测试集 ----------
idx = randperm(numInst);
numTrain = 100;                   % 训练样本数量
numTest = numInst - numTrain;     % 测试样本数量
trainData = data(idx(1:numTrain),:);  
testData = data(idx(numTrain+1:end),:);
trainLabel = labels(idx(1:numTrain)); 
testLabel = labels(idx(numTrain+1:end));

% 标准化数据（使用训练集的均值和标准差）
mean_train = mean(trainData);
std_train = std(trainData);
trainData = (trainData - ones(size(trainData,1),1) * mean_train) ./ (ones(size(trainData,1),1) * std_train);
testData = (testData - ones(size(testData,1),1) * mean_train) ./ (ones(size(testData,1),1) * std_train);

% ---------- 训练 SVM 模型 ----------
model = cell(numLabels,1);
for k=1:numLabels
    model{k} = svmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
end

% ---------- 测试 SVM 模型 ----------
prob = zeros(numTest, numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
    prob(:,k) = p(:, model{k}.Label==1);    %# 概率
end

% ---------- 预测类别 ----------
[~,pred] = max(prob,[],2);

% ---------- 分类结果评价 ----------
acc = sum(pred == testLabel) / numel(testLabel);    % 准确率
disp(['Accuracy: ', num2str(acc)]);

% 混淆矩阵
C = confusionmat(testLabel, pred);
disp('Confusion Matrix:');
disp(C);

% 计算评价指标：精度（Precision）、召回率（Recall）、F1 分数（F1-score）
precision = diag(C) ./ sum(C, 1)';  % 按列计算 Precision
recall = diag(C) ./ sum(C, 2);      % 按行计算 Recall
F1 = 2 * (precision .* recall) ./ (precision + recall);  % F1-score

disp('Precision for each class:');
disp(precision);
disp('Recall for each class:');
disp(recall);
disp('F1-score for each class:');
disp(F1);

% ---------- 可视化结果 ----------
% 比较预测结果与真实结果
figure;
subplot(1, 2, 1);
gscatter(testData(:,1), testData(:,2), testLabel, 'rgb', 'osd');
title('True Labels');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');

subplot(1, 2, 2);
gscatter(testData(:,1), testData(:,2), pred, 'rgb', 'osd');
title('Predicted Labels');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3');
