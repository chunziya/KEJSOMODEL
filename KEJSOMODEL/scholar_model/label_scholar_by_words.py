import json
import mysql.connector

load_f = open("data/fields_word_old.json", 'r')
load_dict = json.load(load_f)


def find(aList, num):
    result = dict()
    length = len(aList)
    for k in load_dict.keys():
        result[k] = len([i for i in aList if i in load_dict[k]])
    if max(result.values()) == 0:
        return None, None
    else:
        L = sorted(result.items(), key=lambda item: item[1], reverse=True)
        L = L[:num]
        return [l[0] for l in L], ['{:.3f}'.format(l[1]/length) for l in L]


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

sql = "select id,fields from scholar where fields is not null and fields != '' limit 0,20"
cursor.execute(sql)
values = cursor.fetchall()
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
        update_sql = "update scholar set label = '%s',label_weight = '%s' where id = '%s'" % (",".join(label), ",".join(label_weight), scholar_id)
        update_cursor.execute(update_sql)
        update_con.commit()
        update_cursor.close()
        update_con.close()

con.close()
