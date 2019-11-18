from gensim.models import fasttext
from gensim.models import word2vec
import pandas as pd
import numpy as np
import logging
import jieba
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print('data loading...')
data_file = open('data/data_5000/abstract_5000.json', 'r', encoding='utf-8')
data_set = np.array(json.load(data_file))
sentence = [item for item in data_set[:, 1] if item != '']

sens_list = [jieba.lcut(sen, cut_all=False) for sen in sentence]
print(sens_list[0])
print('data load succeed!')

model = word2vec.Word2Vec(sens_list, min_count=1, iter=20)
model.save("out/word2vec_5000.model")
