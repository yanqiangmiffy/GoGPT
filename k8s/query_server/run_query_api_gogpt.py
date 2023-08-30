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
import logging

import torch
import uvicorn
from fastapi import FastAPI, Request
from loguru import logger
from transformers import LlamaForCausalLM, LlamaTokenizer

logger.add('/model/output-gogpt2-3b-query/logs/query_api.log')


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


app = FastAPI()

prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
)

model_name = '/model/output-gogpt2-3b-query'
tokenizer = LlamaTokenizer.from_pretrained(model_name,
                                           trust_remote_code=True)
print(tokenizer)
model = LlamaForCausalLM.from_pretrained(model_name,
                                         trust_remote_code=True, device_map='auto').half()
model.eval()


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

    input_text = generate_prompt(instruction=raw_input_text)
    print(f"{input_text}")
    inputs = tokenizer(input_text, return_tensors="pt")

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda'),
        max_new_tokens=1024,
        temperature=0.95,
        # do_sample=True,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.1,
    )
    generation_output = model.generate(bos_token_id=tokenizer.bos_token_id,
                                       eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.pad_token_id,
                                       **generate_kwargs)
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
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
    uvicorn.run(app="run_query_api_gogpt:app", host='0.0.0.0', port=8893)
