import re
import json
import requests
import mysql.connector

'''
生成用于模型训练的数据特点
    1. 使用表czc_classification_czc
    2. 期刊综合——非必须
    3. 样本均衡——必须
    4. 按年份统计——非必须
    5. 摘要和题目
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


f = open('out/abstract_5000.json', 'w', encoding='utf-8')
train_data = []

sql = "select * from czc_classification_czc where journal is not null"
cursor.execute(sql)
values = cursor.fetchall()

for sub in values:
    i = 0
    print(sub['first'] + " is collecting...")
    # dict_trainbag[sub['first']] = []
    journal_list = sub['journal'].split('/')
    print(journal_list)
    for year in range(1999,2019):
        for journal in journal_list:
            sql_word = "select title_cn,keyword_cn,brief_cn from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
            cursor_1.execute(sql_word)
            word_values = cursor_1.fetchall()
            for item in word_values:
                i = i + 1
                title = re.sub(r'[^\w\s]', "", item['title_cn']).replace(" ", "")
                brief = item['brief_cn']
                if brief is not None:
                    train_data.append([title, brief, sub['first']])
            if i > 5000:
                break
        if i > 5000:
                break

json.dump(train_data, f, indent=4, separators=(',', ': '), ensure_ascii=False)
con.close()
f.close()
