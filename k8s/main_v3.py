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
app = FastAPI()

gogpt2_tokenizer = LlamaTokenizer.from_pretrained("/home/searchgpt/pretrained_models/gogpt2-7b-v4.4",
                                                  trust_remote_code=True)
gogpt2_model = LlamaForCausalLM.from_pretrained("/home/searchgpt/pretrained_models/gogpt2-7b-v4.4",
                                                trust_remote_code=True, device_map='auto').half()
gogpt2_model.eval()


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()



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
async def create_item(request: Request):
    global gogpt2_model, gogpt2_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    input_text = generate_prompt(instruction=raw_input_text)
    inputs = gogpt2_tokenizer(input_text, return_tensors="pt")

    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda'),
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
    uvicorn.run(app, host='0.0.0.0', port=8890, workers=1)
