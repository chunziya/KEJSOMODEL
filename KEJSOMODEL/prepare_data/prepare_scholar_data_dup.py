import re
import json
import requests
import mysql.connector

'''
生成用于学者匹配的数据特点
    1. 使用表czc_classification_czc
    2. 期刊综合——非必须
    3. 样本均衡——非必须
    4. 按年份统计——非必须
    5. 关键字（去重）
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

f = open('out/fields_word_dup.json', 'w', encoding='utf-8')
dict_wordbag = dict()

sql = "select * from czc_classification_czc where journal is not null"
cursor.execute(sql)
values = cursor.fetchall()

for sub in values:
    i = 0
    print(sub['first'] + " is collecting...")
    dict_wordbag[sub['first']]=[]
    journal_list = sub['journal'].split('/')
    print(journal_list)
    for year in range(1999,2019):
        for journal in journal_list:
            sql_word = "select keyword_cn from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
            cursor_1.execute(sql_word)
            word_values = cursor_1.fetchall()
            for item in word_values:
                keyword = re.sub(r'[^||\w\s]', "||", item['keyword_cn'])
                if keyword is not None:
                    word_list = list(filter(None, keyword.split("||")))
                    dict_wordbag[sub['first']].extend(word_list)

    # dict_wordbag[sub['first']] = list(set(dict_wordbag[sub['first']]))

json.dump(dict_wordbag, f, indent=4, separators=(',', ': '), ensure_ascii=False)

con.close()
f.close()
