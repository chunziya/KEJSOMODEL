import re
import json
import requests
import mysql.connector

# f = open('journal_paper_count.txt','w')
config = {
    'user': 'user1012',
    'password': '123456',
    'host': '192.168.229.151',
    'database': 'kejso',
    'charset': 'utf8',
    "use_pure": True
}
con = mysql.connector.connect(**config)
journal_cursor = con.cursor(dictionary=True)

first = '图书馆、情报与档案管理'
sql_journal = "select journal from czc_classification where first= '"+ first +"'"
journal_cursor.execute(sql_journal)
values = journal_cursor.fetchall()
for item in values:
#     f.write(item['first'] + '\n')
    print(first + ":")
    journal_list = item['journal'].split('/')
    for journal in journal_list:
        count = 0
        for year in range(1999,2019):
            sql_word = "select count(*) from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
            count_con = mysql.connector.connect(**config)
            cursor = count_con.cursor(dictionary=True)
            cursor.execute(sql_word)
            n = cursor.fetchall()
            # print("czc_journal_all_" + str(year) + ':' + str(n[0]['count(*)']))
            count = count + n[0]['count(*)']
        # f.write(journal + "共" + str(count) + "条数据\n")
        print(journal + "共" + str(count) + "条数据")


# count = 0
# journal = '计算机学报'
# for year in range(1999,2019):
#     sql_word = "select count(*) from czc_journal_all_" + str(year) + " where journal_cn = '" + journal + "'"
#     count_con = mysql.connector.connect(**config)
#     cursor = count_con.cursor(dictionary=True)
#     cursor.execute(sql_word)
#     n = cursor.fetchall()
#     print("czc_journal_all_" + str(year) + ':' + str(n[0]['count(*)']))
#     count = count + n[0]['count(*)']
# print(journal + "共" + str(count) + "条数据")

count_con.close()
# f.close()
