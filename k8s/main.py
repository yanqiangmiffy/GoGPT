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
from transformers import LlamaForCausalLM, LlamaTokenizer
app = FastAPI()


DEVICE = "cuda"
CHATGLM_DEVICE_ID = "0"
CHATGLM_CUDA_DEVICE = f"{DEVICE}:{CHATGLM_DEVICE_ID}"



GOGPT_DEVICE_ID = "1"
GOGPT_CUDA_DEVICE = f"{DEVICE}:{GOGPT_DEVICE_ID}"


chatglm_tokenizer = AutoTokenizer.from_pretrained("/model/chatglm-6b", trust_remote_code=True)
chatglm_model = AutoModel.from_pretrained(
    pretrained_model_name_or_path="/model/chatglm-6b",
    trust_remote_code=True,
    # device_map='auto'
).half().to(CHATGLM_CUDA_DEVICE)
chatglm_model.eval()
print("chatglm加载成功")



gogpt2_tokenizer = LlamaTokenizer.from_pretrained("/model/gogpt2-7b",
                                                  trust_remote_code=True)
gogpt2_model = LlamaForCausalLM.from_pretrained(
    "/model/gogpt2-7b",
    trust_remote_code=True,
    # device_map='auto'
).half().to(GOGPT_CUDA_DEVICE)

gogpt2_model.eval()
print("gogpt2加载成功")




def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


@app.post("/chatglm")
async def create_chatglm(request: Request):
    global chatglm_model, chatglm_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    # history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = chatglm_model.chat(chatglm_tokenizer,
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
    log = "chatglm==>[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer





prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)


def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


@app.post("/gogpt")
async def create_gogpt(request: Request):
    global gogpt2_model, gogpt2_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    input_text = generate_prompt(instruction=raw_input_text)
    inputs = gogpt2_tokenizer(input_text, return_tensors="pt")

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to(GOGPT_CUDA_DEVICE),
        max_new_tokens=4096,
        temperature=0.6,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
    )
    generation_output = gogpt2_model.generate(**generate_kwargs)
    s = generation_output[0]
    output = gogpt2_tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### Response:")[1].strip()

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', 'GoGPT').replace('chatglm-6b', 'GoGPT'),
        "status": 200,
        "time": time
    }
    log = "gogpt===>[" + time + "] " + '", prompt:"' + raw_input_text + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    uvicorn.run(app="main:app", host='0.0.0.0', port=8090, workers=1)
