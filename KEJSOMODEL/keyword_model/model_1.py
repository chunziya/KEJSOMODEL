from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, BatchNormalization, concatenate, Lambda
from keras.models import Model, Sequential, load_model
from keras import optimizers
import warnings
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告

def myLSTM(vocab_size, max_sequence_length, embedding_size=128, hidden_size=64, dropout=0.2):
    model = Sequential()
    model.add(Embedding(vocab_size + 1, embedding_size, input_length = 15))
    model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))
    model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(78, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

    # history = model.fit(x_train, y_train, epochs=15, batch_size=400, validation_split = 0.2)
    # model.save("/Users/czc/PythonFile/fields-model-bilstm-2.h5")
