import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 数据加载和预处理
df = pd.read_csv('./S4248SM144NCEN.csv', index_col='DATE', parse_dates=True)
df.index.freq = 'MS'
df.columns = ['Sales']
df.plot(figsize=(16, 8))

train = df.iloc[:316]
test = df.iloc[316:316 + 12]

scaler = MinMaxScaler()
scaler.fit(train)

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 12
n_feature = 1

train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# 构建模型
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_input, n_feature), return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()

# 训练模型
model.fit(train_generator, epochs=50)

# 绘制损失曲线
loss = model.history.history['loss']
plt.plot(range(len(loss)), loss)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
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

# 绘制预测结果
test.plot(figsize=(12, 8))
plt.show()
