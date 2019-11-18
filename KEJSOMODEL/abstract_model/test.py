import json
import jieba
import numpy as np
from gensim.models import word2vec

print("jieba测试")
string = "本刊讯 1月6日上午,北京市档案工作会议隆重举行,国家档案局副局长、中央档案馆副馆长毛福民和中共北京市委常委、秘书长杜德印两位领导同志到会发表了重要讲话.会议由市政府秘书长刘志华同志主持."
print(jieba.lcut(string))

# print("word2vec测试")
# word = '兽医'
# fname = 'out/word2vec/word2vec.model'
# model = word2vec.Word2Vec.load(fname)
# print(word + ' 相关词：')
# print(model.wv.most_similar(word))

print("摘要统计")
abstract_dict = {}
f = open("data/data_5000/abstract_5000.json")
data_set = np.array(json.load(f))
labels = list(set(data_set[:, 2]))
for i in labels:
    abstract_dict[i] = 0
for item in data_set:
    if item[1] != "":
        abstract_dict[item[2]] = abstract_dict[item[2]]+1
print(abstract_dict)
# json.dump(abstract_dict, open("out/abstract.json", 'w', encoding='utf-8'), separators=(',', ': '), indent=4,
#           ensure_ascii=False)
