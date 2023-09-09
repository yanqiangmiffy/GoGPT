import json
import time
import requests
import pandas as pd
questions = ['长期不吃晚餐真的会瘦吗',
             '为什么最后 1% 的进度条很难加载？',
             '历史上真实可考的女将和女性军事统帅有哪些？',
             '合成步兵营与现在的普通步兵营有什么差别？',
             '线性代数的发明过程和意义是什么？',
             '我认为愚公是真的愚蠢，大家怎么看？',
             '为什么虾和昆虫同属节肢动物，而人类大都喜欢吃虾肉却不喜欢吃虫肉？',
             '人类至今有没有研发出人畜无害的老鼠药？为什么？',
             '有没有化学大佬可以给我讲一讲核外电子排布？',
             '为什么商朝和不少文明会盛行人祭，觉得以人祭祀，可以取悦神明？',
             '瞎扯 · 如何正确地吐槽',
             '人是这个地球上生活得最艰难的生物吗？',
             '美商务部长雷蒙多称美将继续对华出售芯片，但不卖最顶尖芯片，释放了哪些信号？',
             '日本正式启动核污染水排海，我国已组织开展海洋辐射环境监测，海鲜还能吃吗？如何保证涉日食品安全？',
             '教育部新出台的 101 计划会逐渐代替 985，211 吗？',
             '伊甸园字幕宣布解散，称电影《孤注一掷》凭臆想将赌博甩给自己，具体情况如何，怎样看待此事？',
             '中国房地产报刊文，呼吁允许开发商以降价促销的方式展开自救，对此如何解读？',
             '塞尔维亚男篮球员遭对手肘击后，因菲律宾医院无法找到供血，无奈摘除肾脏，如何看待这一后果？',
             'iPhone 15 全系国行售价曝光，你对该系列机型有何期待？对比华为 Mate60 你买谁？',
             '山东发文要求全面落实带薪休假制度，鼓励错峰休假，弹性作息，促进假日消费，将产生哪些影响？']

df= []
for question in questions:
    data = {
        "prompt": question
    }

    post_json = json.dumps(data)
    start_time = time.time()
    r2 = requests.post("http://172.16.120.23:31280/gogpt", data=post_json)
    end_time = time.time()

    time_cost = end_time - start_time
    print(r2.text)
    response = r2.json()['response']
    df.append({'question': question, 'answer': response, 'time_cost': time_cost})

pd.DataFrame(df).to_csv('llm_service_speed_local_v2.csv', index=False)
