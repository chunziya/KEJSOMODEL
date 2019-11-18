import re
import json
import jieba
import requests
import mysql.connector


# 生成计算机领域的年度词袋
#   1. 以期刊《计算机应用研究》为例
#   2. 按年份统计
#   3. 关键字组和title

config = {
    'user': 'user1012',
    'password': '123456',
    'host': '192.168.229.151',
    'database': 'kejso',
    'charset': 'utf8',
    "use_pure": True
}
con = mysql.connector.connect(**config)
cursor_1 = con.cursor(dictionary=True)

f = open('data/computer_science_by_year.json', 'w', encoding='utf-8')
dict_textbag = dict()
journal_list = ['计算机学报','计算机应用研究']
# stop_words = []
stop_words = [line.strip() for line in open('../stopwords.txt', 'r', encoding='utf-8').readlines()]
jieba.add_word("C/S")
jieba.add_word("B/S")

for year in range(1999,2019):
    dict_textbag[year] = []
    for journal in journal_list:       
        sql_word = "select title_cn,keyword_cn from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
        cursor_1.execute(sql_word)
        word_values = cursor_1.fetchall()
        for item in word_values:
            # keyword = item['keyword_cn']
            # title_cn = re.sub(r'[^/\w\s]', "", item['title_cn'])
            keyword = re.sub(r'[^||/\w\s]', "||", item['keyword_cn'])
            if keyword is not None:
                word_list = list(filter(None, keyword.split("||")))
                # title_words = list(filter(None, jieba.lcut(title_cn, cut_all=False)))
                # for word in title_words:
                #     if word not in stop_words and word not in word_list:
                #         word_list.append(word)
                keywordstr = " ".join(word_list)
                if len(keywordstr) > 0:
                    dict_textbag[year].append(word_list)

json.dump(dict_textbag, f, indent=4, separators=(',', ': '), ensure_ascii=False)

con.close()
f.close()
