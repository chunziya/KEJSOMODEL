from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, BatchNormalization, concatenate, Lambda
from keras.models import Model, Sequential, load_model
from keras import optimizers
import warnings
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告

def myLSTM(vocab_size, max_sequence_length, embedding_size=128, hidden_size=64, dropout=0.2):
    embedding_layer = Embedding(vocab_size + 1, embedding_size, input_length = 15)
    lstm_layer_1 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
    lstm_layer_2 = LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False)

    dense_1 = Dense(256, activation='relu')
    dense_2 = Dense(128, activation='relu')

    keyword_input = Input(shape=(max_sequence_length,), dtype='int32')
    y1 = lstm_layer_1(embedding_layer(keyword_input))
    y1 = lstm_layer_2(y1)

    tfidf_input = Input(shape=(15, ), dtype='float32')
    y2 = dense_1(tfidf_input)
    y2 = dense_2(y2)

    y = BatchNormalization()(concatenate([y1, y2]))
    y = Dense(256)(y)
    y = Dense(128)(y)
    preds = Dense(78, activation='softmax')(y)

    model = Model(inputs=[keyword_input, tfidf_input], outputs=preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

    # history = model.fit(x_train, y_train, epochs=15, batch_size=400, validation_split = 0.2)
    # model.save("/Users/czc/PythonFile/fields-model-bilstm-2.h5")
