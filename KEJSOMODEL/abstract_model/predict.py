from keras.models import model_from_yaml
from keras.models import load_model
from keras.preprocessing import sequence
from gensim.models import word2vec
from gensim import corpora
import mysql.connector
import numpy as np
import jieba
import json
import yaml
import warnings

warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告

print('loading labels_dict......')
labels_dict = {}
l_dict = json.load(open("out/labels_dict.json", 'r'))
for i,k in enumerate(l_dict.keys()):
    labels_dict[i]=k
print(labels_dict)

print('loading model......')
# with open('out/lstm.yml', 'r') as f:
#     yaml_string = yaml.load(f)
# lstm_model = model_from_yaml(yaml_string)
# print('loading weights......')
# lstm_model.load_weights('out/lstm.h5')
lstm_model=load_model('out/czc_model_40.h5')
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('loading word2vec_model......')
# word2vec_model=word2vec.Word2Vec.load('../../dgcnn/wordvector/word2vec_baike')
word2vec_model=word2vec.Word2Vec.load('out/word2vec/word2vec.model')
gensim_dict = corpora.Dictionary()
gensim_dict.doc2bow(word2vec_model.wv.vocab.keys(), allow_update=True)
w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过1的词语的索引

def parse_dataset(combined):
    data = []
    for sentence in combined:
        new_txt = []
        for word in sentence:
            new_txt.append(w2indx.get(word,0))
        data.append(new_txt)
    return data


def create_dictionaries(model=None, combined=None, max_len = 500):
    if (combined is not None) and (model is not None):
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=max_len)  # 每个句子所含词语对应的索引，所以句子中含有频数小于1的词语，索引为0
        return combined
    else:
        print('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    combined=create_dictionaries(word2vec_model,words)
    return combined


def lstm_predict(string):
    data=input_transform(string)
#     print(data)
    result=lstm_model.predict_classes(data)
    print(result)
    print(labels_dict[result[0]])

config = {
    'user': 'user1012',
    'password': '123456',
    'host': '192.168.229.151',
    'database': 'kejso',
    'charset': 'utf8',
    "use_pure": True
}
con = mysql.connector.connect(**config)
cursor = con.cursor(dictionary=True)

print("selecting data...")
sql = "select brief_cn from czc_journal_2017 where brief_cn is not null and brief_cn != '' limit 0,100"
# sql = "select experience from scholar where experience is not null and experience != '' limit 0,100"
cursor.execute(sql)
values = cursor.fetchall()
for item in values:
    string=item['brief_cn']
#     string=item['experience']
    print(string)
    lstm_predict(string)