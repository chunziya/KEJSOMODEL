from keras.layers.core import Dense, Dropout
from keras import Input, metrics
from keras.layers.embeddings import Embedding
from keras.layers import *
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
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
hidden_size = 100  # lstm 嵌入的尺寸
batch_size = 512
epoch = 40


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
                    new_txt.append(w2indx.get(word,0))
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


# Attention
import keras.backend as K
class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs

class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)

class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# 加载数据
print('Loading Data and Tokenising...')
combined, y = loadfile()
print(combined.shape, y.shape)
print('Loading a Word2vec model...')
index_dict, word_vectors, combined = word2vec_load(combined)
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
print(x_train.shape, y_train.shape)

# 搭建模型
x_input = Input(shape=(word_dim,))
x = Embedding(n_symbols,
              word_dim,
              mask_zero=False,
              weights=[embedding_weights],
              input_length=max_len,
              trainable=False)(x_input)
#   x = SpatialDropout1D(0.2)(x)
x = Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Bidirectional(CuDNNLSTM(hidden_size, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Attention(8, 16)([x, x, x])
hidden = concatenate([
    GlobalMaxPooling1D()(x),
    GlobalAveragePooling1D()(x),
])
x = Dense(128)(hidden)
y = Dense(69, activation='softmax')(x)
model = Model(x_input,y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 训练模型
model_chechpoint = ModelCheckpoint('out/model_attention.h5', save_best_only=True, save_weights_only=False)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[model_chechpoint])