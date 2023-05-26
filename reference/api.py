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
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from typing import Optional
import requests
import os
DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()



PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def generate_input(instruction: Optional[str] = None, input_str: Optional[str] = None) -> str:
    if input_str is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT['prompt_input'].format_map({'instruction': instruction, 'input': input_str})


def convert_history_to_text(history):

    user_input = history[-1][0]

    text = generate_input(user_input)
    return text






@app.post("/")
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

    prompt=convert_history_to_text(prompt)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(DEVICE)
    outputs = model.generate(input_ids)
    response=tokenizer.decode(outputs[0],skip_special_tokens=True)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B','GoGPT').replace('chatglm-6b','GoGPT'),
        # "history": [[x[0].replace('ChatGLM-6B','GoGPT').replace('chatglm-6b','GoGPT'),x[1].replace('ChatGLM-6B','GoGPT').replace('chatglm-6b','GoGPT')]for x in history],
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/home/searchgpt/yq/GoGPT/output-bloom-3b", trust_remote_code=True)
    model = AutoModel.from_pretrained("/home/searchgpt/yq/GoGPT/output-bloom-3b", trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8889, workers=1)
