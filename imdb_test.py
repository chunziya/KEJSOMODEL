from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print(train_data[0], train_labels[0])
# print(max([max(sequence) for sequence in train_data]))

import numpy as np
# 不能将整数序列直接输入神经网络，需要将列表转换为张量：one-hot 编码
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)  #将训练数据向量化
x_test = vectorize_sequences(test_data)  #将测试数据向量化
y_train = np.asarray(train_labels).astype('float32')  #标签向量化
y_test = np.asarray(test_labels).astype('float32')
# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 初始模型
from keras import models, layers, regularizers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 向模型添加 L2 权重正则化
model1 = models.Sequential()
model1.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model1.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))
model1.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
history1 = model1.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 向 IMDB 网络中添加 dropout
model2 = models.Sequential()
model2.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
history2 = model2.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 绘图
import matplotlib.pyplot as plt
val_loss_values = history.history['val_loss']
val_loss_values1 = history1.history['val_loss']
val_loss_values2 = history2.history['val_loss']
epochs = range(1, len(val_loss_values) + 1)

plt.plot(epochs, val_loss_values2, 'b+', label = 'Dropout model')
plt.plot(epochs, val_loss_values1, 'bo', label = 'L2-regulation model')
plt.plot(epochs, val_loss_values, 'b', label = 'Original model')
plt.title('Contrast')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend() 
plt.show()

# history = model.fit(x_train, y_train, epochs=4, batch_size=512)
# print(history.history.keys())
# results = model.evaluate(x_test, y_test)
# print(results)
