import numpy as np
import sklearn
import logging
import math
import json
import codecs
import pickle
import heapq
from collections import Counter

from keras.models import load_model
from keras.preprocessing.text import text, Tokenizer
from keras.preprocessing.sequence import pad_sequences

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

train_data = json.load(open('../data/fields_train.json', 'r', encoding='utf-8'))
label_dict = json.load(open("../data/fields_label.json", "r", encoding='utf-8'))
print(label_dict.keys())

def trans_label(labelstr):
    return label_dict[labelstr]

# 打乱数据
np.random.seed(200)
np.random.shuffle(train_data)

keywords_list = [p[1] for p in train_data]
# self.tokenizer = pickle.load(open("../data/tokenizer_words", "rb"))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(keywords_list)
vocab_size = len(tokenizer.word_index)


# doc_size = tokenizer.document_count
# word_counts = tokenizer.word_counts  # 保存每个word在所有文档中出现的次数
# sum_word_counts = sum(word_counts.values())
# word_docs = tokenizer.word_docs  # 保存每个word出现的文档的数量


model = load_model('out/model_2(best).h5')

test_data = [
    "信息生产力 劳动者 劳动工具 劳动对象 转向 生产力要素 特征 高科技含量 电子信息技术 电子信息产业 外延和内涵 计算机系统 第一生产力 运用 使用信息 科学技术 经济增长 电子技术 智能化 虚拟性",
    "马克思主义史学 日本 国际政治背景 批判与继承 传播和发展 时代变迁 社会发展 经济状况 分支学科 史学史 生命力 近现代 脉搏 轨迹 地位",
    "柔性关节机械臂 反演控制 李雅普络夫法 轨迹跟踪控制",
    "虚拟人 骨骼模型 树模型 低序体阵列 递归计算",
    "骨质疏松 体层摄影术 X线计算机 骨密度"]
label = ["哲学","历史学","机械工程","计算机科学与技术","临床医学"]

keywords = tokenizer.texts_to_sequences(test_data)
keywords = np.array(pad_sequences(keywords, maxlen=15))
y_pred = model.predict(keywords, batch_size = 1)
for res in y_pred:
    re = res.tolist()
    re1 = map(re.index, heapq.nlargest(5, re))
    print(list(re1))
    # re = map(y_pred.index, heapq.nlargest(3, y_pred))
    # print(re)
print(label)


def caculate_tfidf(words):
    tfidf = []
    seq = text.text_to_word_sequence(words)
    counts = Counter(seq)
    for item in seq:
        tf_idf = float((counts[item]/sum(counts.values())) * math.log(doc_size/(word_docs[item]+1), 10))
        # tf = self.word_counts[item]/self.sum_word_counts
        # idf = math.log(self.doc_size/(self.word_docs[item]+1), 10)
        tfidf.append(tf_idf)
    return tfidf
        