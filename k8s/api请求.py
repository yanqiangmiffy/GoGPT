#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: demo.py
@time: 2023/08/17
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import requests
import json
data = {
    "prompt": "怎么学习机器学习"
}
post_json = json.dumps(data)
r1 = requests.post("http://172.16.120.23:31280/chatglm", data=post_json)
r2 = requests.post("http://172.16.120.23:31280/gogpt", data=post_json)
r3 = requests.post("http://172.16.120.23:31280/chatglm2", data=post_json)
print(r1.text)
print(r2.text)
print(r3.text)
