#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: api.py
@time: 2023/05/11
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: https://github.com/THUDM/ChatGLM-6B/blob/main/api.py
"""
import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

app = FastAPI()


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@app.post("/chatglm")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    # history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=[],
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/model/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path="/model/chatglm-6b",
        trust_remote_code=True,
        device_map='auto').half()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8090, workers=1)
