import numpy as np
import sklearn
import logging
import math
import json
import codecs
import pickle
from collections import Counter

from keras.preprocessing.text import text, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class TextDataset(object):

    def __init__(self, seed, shuffle, max_sequence_length, batch_size):
        # load data
        logger.info('loading training datas...')
        self.seed = seed
        self.batch_size = batch_size
        self.msl = max_sequence_length
        self.train_data = json.load(open('data/train_keyword.json', 'r', encoding='utf-8'))
        self.label_dict = json.load(open("data/fields_label.json", "r", encoding='utf-8'))
        # 打乱数据
        np.random.seed(200)
        np.random.shuffle(self.train_data)
        print(self.train_data[0])

        self.keywords_list = [p[1] for p in self.train_data]
        # self.tokenizer = pickle.load(open("../data/tokenizer_words", "rb"))
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.keywords_list)
        self.vocab_size = len(self.tokenizer.word_index)

        self.doc_size = self.tokenizer.document_count
        self.word_counts = self.tokenizer.word_counts  # 保存每个word在所有文档中出现的次数
        self.sum_word_counts = sum(self.word_counts.values())
        self.word_docs = self.tokenizer.word_docs  # 保存每个word出现的文档的数量

        print(self.doc_size)
        print(self.sum_word_counts)

        self.keywords = self.tokenizer.texts_to_sequences(self.keywords_list)
        self.labels = [self.trans_label(p[2]) for p in self.train_data]

        self.keywords = pad_sequences(self.keywords, maxlen=self.msl)
        self.labels, self.keywords = np.array(self.labels), np.array(self.keywords)

        print(self.keywords[0])
        print(self.labels[0])

        # self.stop_list = []
        # with codecs.open('stoplist.txt', 'r', 'utf-8') as f:
        #     for word in f.readlines():
        #         self.stop_list.append(word[:-1])

        # self.title = self.tokenizer.texts_to_sequences([p[0] for p in self.train_data])
        # self.tfidf = [self.caculate_tfidf(p[1]) for p in self.train_data]
        # self.tfidf = pad_sequences(self.tfidf, maxlen=self.msl)
        # print(self.tfidf[0])

        logger.info('training data loaded')
        self.n_pairs = len(self.labels)
        logger.info('all pairs count %d', self.n_pairs)

    def split_dataset(self, test_size):
        train = {'keywords': None, 'labels': None}
        test = train.copy()
        train['keywords'], test['keywords'], train['labels'], test['labels'] = train_test_split(self.keywords,  self.labels, test_size=test_size, random_state=self.seed)
        return DataLoader(self.batch_size, train), DataLoader(self.batch_size, test)

    def trans_label(self, labelstr):
        return self.label_dict[labelstr]

    def caculate_tfidf(self, words):
        tfidf = []
        seq = text.text_to_word_sequence(words)
        counts = Counter(seq)
        for item in seq:
            tf_idf = float((counts[item]/sum(counts.values())) * math.log(self.doc_size/(self.word_docs[item]+1), 10))
            # tf = self.word_counts[item]/self.sum_word_counts
            # idf = math.log(self.doc_size/(self.word_docs[item]+1), 10)
            tfidf.append(tf_idf)
        return tfidf
        
    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return self.keywords[idx], self.labels[idx]


class DataLoader(object):
    def __init__(self, batch_size, data: dict):
        self.batch_size = batch_size
        self.data = data

    def __iter__(self):
        N = len(self.data['labels'])
        iters = N // self.batch_size + 1
        for i in range(iters):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, N)
            yield np.array(self.data['keywords'][start:end]), np.array(self.data['labels'][start:end])
