import re
import json
import numpy as np
import mysql.connector

load_f = open("data/tfidf.json", 'r')
load_dict = json.load(load_f)


def find(aList, num):
    result = dict()
    for k, v in load_dict.items():
        # result[k] = len([i for i in aList if i in load_dict[k]])
        result[k] = np.sum([v.get(i, 0.0) for i in aList])
    if max(result.values()) == 0:
        return None, None
    else:
        L = sorted(result.items(), key=lambda item: item[1], reverse=True)
        L = L[:num]
        return [l[0] for l in L], ['{:.3f}'.format(l[1]) for l in L]


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
sql = "select id,fields from scholar_tfidf where fields is not null and fields != '' limit 0,100"
cursor.execute(sql)
values = cursor.fetchall()
print("labeling start...")
for item in values:
    scholar_id = item['id']
    keyword_list = item['fields'].strip(";").split(";")
    label, label_weight = find(keyword_list, 3)

    # print(label_weight)

    if label is None:
        continue
    else:
        print(label)
        update_con = mysql.connector.connect(**config)
        update_cursor = update_con.cursor()
        update_sql = "update scholar_tfidf set label = '%s',label_weight = '%s' where id = '%s'" % (",".join(label), ",".join(label_weight), scholar_id)
        update_cursor.execute(update_sql)
        update_con.commit()
        update_cursor.close()
        update_con.close()

con.close()
