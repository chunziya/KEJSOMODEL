from keras.callbacks import ModelCheckpoint
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D
from keras.layers.core import Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
from gensim import corpora
import numpy as np
import jieba
import json
import yaml

import warnings

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告

'''
set parameters:
'''
word_dim = 100  # 由于将word2vec训练好的参数直接传入emmbedding层使用，所以此处作为embedding的output_dim与word2vec的词向量维度相同
max_len = 500  # 每句的最大次数
hidden_size = 200  # lstm 嵌入的尺寸
batch_size = 512
epoch = 10


# def to_one_hot(labels, dimension=69):
#     results = dict()
#     # results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[label] = np.zeros(dimension)
#         results[label][i] = 1.
#     return results


# 加载训练文件
def loadfile():
    f = open("data/data_5000/abstract_5000.json")
    data_set = np.array(json.load(f))
    labels_dict = json.load(open("out/labels_dict.json", 'r'))
    # 训练数据分词
    combined = [jieba.lcut(document) for document in data_set[:, 1]
                if document != ""]
    # 标签向量化
    y = [labels_dict[a[2]] for a in data_set if a[1] != ""]
    return np.array(combined), np.array(y)



def create_dictionaries(model=None, combined=None):
    '''
    Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = corpora.Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过1的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过1的词语的词向量

        def parse_dataset(combined):
            '''
            Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=max_len)  # 每个句子所含词语对应的索引，所以句子中含有频数小于1的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_load(combined):
    fname = 'out/word2vec/word2vec.model'
    model = word2vec.Word2Vec.load(fname)
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, word_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    '''
    定义网络结构
    '''
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(n_symbols,
                        word_dim,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=max_len,
                        trainable=False))  # Adding Input Length
    model.add(SpatialDropout1D(0.4))
    model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    model.add(Dense(69, activation='softmax'))
    model.summary()

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model_chechpoint = ModelCheckpoint('out/czc_model_40.h5', save_best_only=True, save_weights_only=False)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoch,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[model_chechpoint])

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    print('Test score:', score)



# 训练模型，并保存
def train():
    print('Loading Data and Tokenising...')
    combined, y = loadfile()
    print(combined.shape, y.shape)
    print('Loading a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_load(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    train()
