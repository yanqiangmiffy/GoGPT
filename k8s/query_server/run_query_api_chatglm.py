#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: run_query_api_chatglm.py.py
@time: 2023/08/30
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/model/chatglm-6b', trust_remote_code=True)
    model = AutoModel.from_pretrained('/model/chatglm-6b',
                                      trust_remote_code=True)
    model = model.half().cuda()
    model = PeftModel.from_pretrained(model, '/model/query-chatglm-6b-lora')

    model = model.eval()
    return model, tokenizer

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    response, history = model.chat(tokenizer, raw_input_text, history=[],
                                   max_length=2048, top_p=0.7,
                                   temperature=0.95)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        "status": 200,
        "time": time
    }
    log = "===>[" + time + "]\n " + '", prompt:"\n' + raw_input_text + '", response:"\n' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    model, tokenizer=load_model()
    uvicorn.run(app="run_query_api_chatglm:app", host='0.0.0.0', port=8893)
