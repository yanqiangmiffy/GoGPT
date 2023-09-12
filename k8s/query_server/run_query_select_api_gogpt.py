#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: api.py
@time: 2023/07/10
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from loguru import logger
from transformers import LlamaForCausalLM, LlamaTokenizer

print(torch.cuda.is_available())  # cuda是否可用
print(torch.cuda.device_count())  # gpu数量
print(torch.cuda.current_device())  # 当前设备索引, 从0开始
print(torch.cuda.get_device_name(0))  # 返回gpu名字

logger.add('/model/logs/query_select_api.log')


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


app = FastAPI()

prompt_input = (
    "{instruction}\n\n### Response:\n\n"
)


def load_query_model():
    global query_model, query_tokenizer
    model_name = '/model/output-gogpt2-7b-query-question-0906-2'
    query_tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(query_tokenizer)
    query_model = LlamaForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16).half()
    query_model.eval()
    query_model.to('cuda:0')
    return query_model, query_tokenizer


query_model, query_tokenizer = load_query_model()
logger.info("加载query模型成功")


def load_select_model():
    global select_model, query_tokenizer
    model_name = '/model/output-gogpt2-7b-doc-select-0907'
    select_tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(select_tokenizer)
    select_model = LlamaForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16).half()
    select_model.eval()
    select_model.to('cuda:0')
    return select_model, select_tokenizer


select_model, select_tokenizer = load_select_model()
logger.info("加载select模型成功")


def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


@app.post("/query")
async def create_query_item(request: Request):
    global query_model, query_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    input_text = generate_prompt(instruction=raw_input_text)
    logger.info(f"{input_text}")
    inputs = query_tokenizer(input_text, return_tensors="pt")

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda:0'),
        max_new_tokens=1024,
        temperature=0.95,
        # do_sample=True,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
    )
    generation_output = query_model.generate(bos_token_id=query_tokenizer.bos_token_id,
                                             eos_token_id=query_tokenizer.eos_token_id,
                                             pad_token_id=query_tokenizer.pad_token_id,
                                             **generate_kwargs)
    s = generation_output[0]
    output = query_tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### Response:")[1].strip()

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        "status": 200,
        "time": time
    }
    log = "===>[" + time + "]\n " + '", prompt:"\n' + raw_input_text + '", response:"\n' + repr(response) + '"'
    logger.info(log)
    torch_gc()
    return answer


@app.post("/select")
async def create_select_item(request: Request):
    global select_model, select_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    input_text = generate_prompt(instruction=raw_input_text)
    logger.info(f"{input_text}")
    inputs = select_tokenizer(input_text, return_tensors="pt")

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda:0'),
        max_new_tokens=1024,
        temperature=0.95,
        # do_sample=True,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
    )
    generation_output = select_model.generate(bos_token_id=select_tokenizer.bos_token_id,
                                              eos_token_id=select_tokenizer.eos_token_id,
                                              pad_token_id=select_tokenizer.pad_token_id,
                                              **generate_kwargs)
    s = generation_output[0]
    output = select_tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### Response:")[1].strip()

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        "status": 200,
        "time": time
    }
    log = "===>[" + time + "]\n " + '", prompt:"\n' + raw_input_text + '", response:"\n' + repr(response) + '"'
    logger.info(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run(app="run_select_api_gogpt:app", host='0.0.0.0', port=8893)
