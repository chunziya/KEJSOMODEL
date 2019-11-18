import json
import pickle
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

load_f = open("../data/fields_word_dup_old.json", 'r')
load_dict = json.load(load_f)

texts_list = []
# sub_count = dict()
for k, v in load_dict.items():
    # sub_count[k] = len(v)
    texts_list.append(" ".join(v))
# print(sub_count)

# 创建label字典
# label_dict = dict()
# n = 0
# for k in load_dict.keys():
#     label = [0] * 78
#     label[n] = 1
#     label_dict[k] = label
#     n = n + 1

# json.dump(label_dict, open("data/fields_label.json", 'w', encoding='utf-8'),
          # separators=(',', ': '), indent=4, ensure_ascii=False)

# # 创建分词器
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts_list)
# pickle.dump(tokenizer, open("领域分类/keyword/tokenizer_words", "wb"))


cv = CountVectorizer()  # 创建词袋数据结构
cv_fit = cv.fit_transform(texts_list)

# # 列表形式呈现文章生成的词典--词集
word_set = cv.get_feature_names()
print("文本关键字：")
length = len(word_set)
print(length)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(cv_fit)
tfidf_list = tfidf.toarray()

# # 关键词过滤
# n = 0
# flag = 0
sub_list = list(load_dict.keys())
tfidf_dict = dict()
for i in range(tfidf_list.shape[0]):
    print(sub_list[i])
    w_l = [(word_set[j], tfidf_list[i][j]) for j in range(length) if tfidf_list[i][j] > 0.0]
    tfidf_dict[sub_list[i]] = dict(w_l)
json.dump(tfidf_dict, open("tfidf_old.json", 'w', encoding='utf-8'), separators=(',', ': '), indent=4, ensure_ascii=False)

# 字典形式呈现，key：词，value:索引
# print(cv.vocabulary_	)       
# print(cv_fit)
# （0,3） 1   第0个列表元素，**词典中索引为3的元素**，当前文档中的词频
# （0,1） 1
# （0,2） 1
# print(cv_fit.toarray()) # 将结果转化为稀疏矩阵矩阵的表示方式

# 每个词在所有文档中的词频，与词集对应
# word_fre = cv_fit.toarray().sum(axis=0)
# print(word_fre)
