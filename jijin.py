#    Author:  a101269
#    Date  :  2020/3/30


import sys
import os
import pycurl
from io import StringIO
import json
import pycurl
from io import BytesIO


import pycurl
from io import BytesIO

c = pycurl.Curl()


codes=[('161022',2998.6,1.3340),('007254',1950.51,1.2817),('162412',2328.02,1.2886),('001052',3237.6,0.6177),('000051',1465.59,1.3646)]   # 501090 无
fene=[2998.6]
change_zonge_tom=0
change_zonge_his=0
for code in codes:
    url='https://fundgz.1234567.com.cn/js/'+code[0]+'.js?rt=1463558676006'
    c.setopt(c.URL,url)
    buffer = BytesIO()
    c.setopt(c.WRITEDATA, buffer)
    c.perform()

    # Decode the response body:
    ret = buffer.getvalue().decode('utf-8')
    ret=ret.strip('jsonpgz(')
    ret=ret.strip(');')
    ret=eval(ret)
    # print(ret)
    print('--------------'+ret['name'])
    # print(ret['dwjz'])  # 当前净值
    # print(float(ret['gsz']))
    print("基准净值：{:.4f}".format(code[2])) # 基准净值
    print("实时净值："+ret['gsz'])# 估算值
    print(ret['gszzl'])  # 估算增长率
    jzzj=float(ret['gsz'])-float(ret['dwjz'])
    zonge_tom=float(ret['gsz'])*code[1]
    zonge_tod=float(ret['dwjz'])*code[1]
    zonge_sta=code[2]*code[1]
    change= zonge_tom - zonge_tod
    change_year=zonge_tom-zonge_sta

    print("今日增减：{:.4f}".format(change))
    print("总共增减：{:.4f}".format(change_year))
    change_zonge_tom+=change
    change_zonge_his+=change_year
print("今日共增长：{:.4f}".format(change_zonge_tom))
print("历史共增长：{:.4f}".format(change_zonge_his))









