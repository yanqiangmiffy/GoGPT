import json

import requests

data = {
    "prompt": "以关键词形式重写查询，分析一下马克龙说法国将加大国防支出，推动军事转型"
}
post_json = json.dumps(data)
r1 = requests.post("http://127.0.0.1:8893/", data=post_json)
# r1 = requests.post("http://172.16.120.23:31280/chatglm", data=post_json)
# r2 = requests.post("http://172.16.120.23:31280/gogpt", data=post_json)
print(r1.text)

data = {
    "prompt": "结合代词消解，以关键词形式重写第二轮查询\n第一轮查询：什么是一揽子化债方案？。第一轮答案：一揽子化债方案是指针对地方债务风险，制定和实施的一系列综合性措施和政策。该方案旨在有效防范和化解地方债务风险，促进地方经济的稳定和可持续发展。\n具体的一揽子化债方案的内容和措施尚未公开，但根据报道和研究报告，该方案可能包括以下方面：\n地方债务置换：通过发行新的债券来置换原有的地方债务，以降低债务成本和优化债务结构。\n金融资源调动：调动政策性银行和国有商业银行等金融资源，支持地方债务的偿还和再融资。\n国有资产盘活：通过盘活国有资产，将一部分资产用于偿还地方债务，减轻地方政府的债务负担。\n特殊再融资债券发行：可能会推出特殊再融资债券，以提供额外的融资渠道，帮助地方政府解决债务问题。\n需要注意的是，以上是关于一揽子化债方案的一些信息，具体的方案内容可能还有其他方面的内容。由于搜索结果有限，可能还有其他相关信息未被涵盖。 第一轮重写后的查询：一揽子化债方案#一揽子化债方案 定义#一揽子化债方案 解释。第二轮查询：它的主要内容是什么？"
}
post_json = json.dumps(data)
r1 = requests.post("http://127.0.0.1:8893/", data=post_json)
# r1 = requests.post("http://172.16.120.23:31280/chatglm", data=post_json)
# r2 = requests.post("http://172.16.120.23:31280/gogpt", data=post_json)
print(r1.text)
