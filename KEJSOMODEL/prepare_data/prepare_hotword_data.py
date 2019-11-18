import re
import json
import requests
import mysql.connector

'''
生成用于年度热词的数据特点
    1. 使用表czc_classification_wzh
    2. 期刊综合——必须
    3. 样本均衡——非必须
    4. 按年份统计——必须
    4. 关键字组
'''

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
cursor_1 = con.cursor(dictionary=True)

f = open('out/fields_text_by_year.json', 'w', encoding='utf-8')
dict_textbag = dict()

sql = "select * from czc_classification_wzh where journal is not null"
cursor.execute(sql)
values = cursor.fetchall()

for sub in values:
    i = 0
    print(sub['first'] + " is collecting...")
    dict_textbag[sub['first']]=dict()
    journal_list = sub['journal'].split('/')
    print(journal_list)
    for year in range(1999,2019):
        dict_textbag[sub['first']][year] = []
        for journal in journal_list:
            sql_word = "select keyword_cn from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
            cursor_1.execute(sql_word)
            word_values = cursor_1.fetchall()
            for item in word_values:
                keyword = re.sub(r'[^||\w\s]', "||", item['keyword_cn'])
                if keyword is not None:
                    word_list = list(filter(None, keyword.split("||")))
                    keywordstr = " ".join(word_list)
                    if len(keywordstr) > 0:
                        dict_textbag[sub['first']][year].append(keywordstr)

json.dump(dict_textbag, f, indent=4, separators=(',', ': '), ensure_ascii=False)

con.close()
f.close()
