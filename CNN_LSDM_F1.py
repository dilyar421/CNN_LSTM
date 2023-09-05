from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten,LeakyReLU,Dropout,Activation
from keras.callbacks import EarlyStopping
from keras import regularizers  # 用于 L1、L2 正则化
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.metrics import Precision, Recall
from keras import backend as K
from keras.optimizers import Adam
# 数据预处理
# 读取数据
train_data = pd.read_csv('train_data.txt', sep="\t")
val_data = pd.read_csv('val_data.txt', sep="\t")
test_data = pd.read_csv('test_data.txt', sep="\t")

train_labels = pd.read_csv("train_data_labels.txt", sep="\t")
val_labels = pd.read_csv("val_data_labels.txt", sep="\t")
test_labels = pd.read_csv("test_data_labels.txt", sep="\t")

# 标签One-Hot编码
encoder = OneHotEncoder(sparse=False)
train_labels = encoder.fit_transform(train_labels)
val_labels = encoder.transform(val_labels)
test_labels = encoder.transform(test_labels)

# 去量纲（Standardization）
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
val_data_scaled = scaler.transform(val_data)
test_data_scaled = scaler.transform(test_data)

# 类别权重
class_weight_dict = {0: 0.0964, 1: 0.5594, 2: 0.1712, 3: 0.1274, 4: 0.0456}

# 模型构建
model = Sequential()

# 1D-CNN 层
model.add(Conv1D(filters=16, kernel_size=2, activation='elu',
                 kernel_regularizer=regularizers.l1_l2(l1=7.608925974926861e-05, l2=3.86763187388361e-06),
                 input_shape=(9, 1)))

# MaxPooling 层
model.add(MaxPooling1D(pool_size=2))

# 额外的 1D-CNN 层
model.add(Conv1D(filters=16, kernel_size=2, padding='valid', strides=1))
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=2))

# LSTM 层
model.add(LSTM(units=65, return_sequences=False, activation='sigmoid',
               kernel_regularizer=regularizers.l1_l2(l1=1.0146752130615344e-06, l2=4.257965569479559e-05),
               dropout=0.17057974716829463, recurrent_dropout=0.1281587271039578))

# Dropout 层
model.add(Dropout(0.008594618186850377))

# 输出层
model.add(Dense(5, activation='softmax'))

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val
# 创建一个Adam优化器实例，并设置学习率
optimizer = Adam(learning_rate=0.0018288429011831756)
# 编译模型
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy', Precision(), Recall(), f1_score])

# 重新塑造数据以适应模型
train_data_scaled = train_data_scaled.reshape((train_data_scaled.shape[0], 9, 1))
val_data_scaled = val_data_scaled.reshape((val_data_scaled.shape[0], 9, 1))
test_data_scaled = test_data_scaled.reshape((test_data_scaled.shape[0], 9, 1))

# 使用早停法防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=6)

# 训练模型
history = model.fit(train_data_scaled, train_labels, epochs=50, batch_size=64, 
                    validation_data=(val_data_scaled, val_labels), 
                    callbacks=[early_stopping], class_weight=class_weight_dict)

# 输出指标
print("Validation Accuracy:", max(history.history['val_accuracy']))
print("Validation Precision:", max(history.history['val_precision']))
print("Validation Recall:", max(history.history['val_recall']))
print("Validation F1 Score:", max(history.history['val_f1_score']))

# 评估模型
loss, accuracy, precision, recall, f1score = model.evaluate(test_data_scaled, test_labels)
print(f"Test loss: {loss}, Test accuracy: {accuracy}, Test Precision: {precision}, Test Recall: {recall}, Test F1 Score: {f1score}")
model.save('my_trained_model_F.h5')