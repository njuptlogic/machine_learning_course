import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 数据加载和预处理
df = pd.read_csv('./S4248SM144NCEN.csv', index_col='DATE', parse_dates=True)
df.index.freq = 'MS'
df.columns = ['Sales']
df.plot(figsize=(16, 8))

# 划分训练集和测试集
train = df.iloc[:316]
test = df.iloc[316:316 + 12]

# 归一化数据
scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 12
n_feature = 1

# 创建训练集生成器
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=16)

# 手动创建验证集
val_split_index = int(len(scaled_train) * 0.8)  # 80% 训练集，20% 验证集
val_data = scaled_train[val_split_index - n_input:]  # 从 val_split_index - n_input 开始
val_targets = scaled_train[val_split_index:]  # 从 val_split_index 开始

# 修正验证集输入长度，确保长度一致
val_data = val_data[: len(val_targets)]

val_generator = TimeseriesGenerator(val_data, val_targets, length=n_input, batch_size=16)

# 构建模型
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(n_input, n_feature), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

model.summary()

# 训练模型
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,  # 使用验证集生成器
    verbose=1
)

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 预测测试数据
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_feature))

for i in range(len(test)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# 反归一化预测结果
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

# 绘制实际值与预测值对比图
plt.figure(figsize=(12, 8))
plt.plot(test.index, test['Sales'], label='Actual Sales', marker='o')
plt.plot(test.index, test['Predictions'], label='Predicted Sales', marker='x')
plt.title("Actual vs Predicted Sales")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.show()


