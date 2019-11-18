import os
import json
import jieba
import logging
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

test = json.load(open(os.path.join('data/data_all', 'abstract1999.json'), 'r', encoding='utf-8'))
print(test[0])


class segment_sen(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(fname)
            sen_list = json.load(open(os.path.join(self.dirname, fname), 'r', encoding='utf-8'))
            for sen in sen_list:
                sen_n = str(sen).replace(' ', '').replace('\n', '').replace('\r', '')
                yield jieba.lcut(sen_n, cut_all=False)


sentences = segment_sen("data/data_all")
model = word2vec.Word2Vec(sentences)  # 默认100维
model.save("out/word2vec/word2vec.model")
