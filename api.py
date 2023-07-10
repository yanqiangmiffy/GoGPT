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
from transformers import LlamaForCausalLM, LlamaTokenizer

DEVICE = "cuda"
DEVICE_ID = "1"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


app = FastAPI()

prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)


def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')
    # history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    # response, history = model.chat(tokenizer,
    #                                prompt,
    #                                history=[],
    #                                max_length=max_length if max_length else 2048,
    #                                top_p=top_p if top_p else 0.7,
    #                                temperature=temperature if temperature else 0.95
    #                                )

    input_text = generate_prompt(instruction=raw_input_text)
    inputs = tokenizer(input_text, return_tensors="pt")
    # generation_output = model.generate(
    #     input_ids=inputs["input_ids"].to('cuda'),
    #     max_new_tokens=2048,
    #     temperature=0.1,
    #     do_sample=True,
    #     top_p=1.0,
    #     top_k=0,
    #     repetition_penalty=1.1,
    # )

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda'),
        max_new_tokens=2048,
        temperature=0.1,
        do_sample=True,
        top_p=top_p,
        top_k=0,
        repetition_penalty=1.1,
    )
    generation_output =model.generate(**generate_kwargs)
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### Response:")[1].strip()
    print("Response: ", response)
    print("\n")

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        # "history": [[x[0].replace('ChatGLM-6B','GoGPT').replace('chatglm-6b','GoGPT'),x[1].replace('ChatGLM-6B','GoGPT').replace('chatglm-6b','GoGPT')]for x in history],
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + raw_input_text + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained("/home/searchgpt/pretrained_models/gogpt-7b-v4", trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained("/home/searchgpt/pretrained_models/gogpt-7b-v4",trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8888, workers=1)
