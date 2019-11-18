import re
import requests
import mysql.connector
from lxml import etree

def catch_data(journal):
    url = "http://thu-ej.cceu.org.cn/cgi-bin/thu?s={}&typ=1"
    req = requests.get(url.format(journal))
    etree_req = etree.HTML(req.text)
    fields = etree_req.xpath('//*[@id="results"]/tr[2]/td[5]/text()')
    issn = etree_req.xpath('//*[@id="results"]/tr[2]/td[2]/text()')
    return fields, issn

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

# 爬取领域28个
# for i in range(1,8):
#     for j in range(1,5):
#         tag = catch_data(i,j)
#         print(tag)
#         cursor.execute("insert into czc_journal_tag28 (tags28,journals) values ('%s', NULL)" %tag)
#         con.commit()

sql = "select id,journal_cn from czc_magazine_journal where standrad_tags is NULL"
# sql = "select id,journal_cn from czc_magazine_journal where journal_cn=\'财贸研究\'"
cursor.execute(sql)
values = cursor.fetchall()
for item in values:
    fields,issn = catch_data(item['journal_cn'])
    print(fields)
    if len(fields) != 0:
        update_sql="update czc_magazine_journal set ISSN = '%s',tags = '%s' where id = %d" %(issn[0],fields[0],item['id'])
        update_cursor = con.cursor()
        update_cursor.execute(update_sql)
        con.commit()
        # print(update_cursor.rowcount)
con.close()