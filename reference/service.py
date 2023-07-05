import gc
import json
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer


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


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    """
        :param model:
        :param tokenizer:
        :param input_texts:
        :return: [[('One', -5.882715702056885),
      (' plus', -9.785109519958496),
      (' one', -0.7229145169258118),
      (' is', -2.494063377380371),
      (' two', -6.137458324432373)],
    ]
    """
    input_ids = tokenizer(input_texts, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1::]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    batch = batch[0]
    logprobs = [x[1] for x in batch]
    top_logprobs_dicts = [{x[0].strip(): x[1]} for x in batch]
    del input_ids, outputs, probs
    torch.cuda.empty_cache()
    return logprobs, top_logprobs_dicts


def convert_tokens(text, tokenizer):
    input_ids = tokenizer(text)['input_ids']
    tokens = []
    for id in input_ids:
        tokens.append(tokenizer.decode(id))
    return tokens

def predict(data,temperature=0.5):
    echo_prompt = False
    try:
        start_time = time.time()
        results = []
        print(data)
        text = PROMPT_DICT["prompt_no_input"].format_map({"instruction": data})
        print(text)
        res_d = []
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        if echo_prompt:
            max_new_tokens = 1
        outputs = model.generate(      
            inputs,
            max_new_tokens=2048,
            top_p=1,
            top_k=0,
            repetition_penalty=1.1,
            temperature=temperature,
        )
        # outputs = model.generate(inputs,max_new_tokens=3)
        # print("请求：长度",max_seq_len)
        for output in outputs:
            print("生产长度:", len(output))
            if echo_prompt:
                output = output[:-1]
                # pass
            else:
                output = output[inputs.size()[1] :]
            print("echo_prompt生成长度:", len(output))
            result = tokenizer.decode(output, skip_special_tokens=True)
            res_d.append(result)
        results.append(res_d)
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        end_time = time.time()
        completions = []
        for idx, sequences in enumerate(results):
            res_d = []
            for result in sequences:
                logprobs, top_logprobs_dicts = to_tokens_and_logprobs(model, tokenizer, ["<s>" + result])
                if not echo_prompt:
                    if len(convert_tokens(result, tokenizer)) == 1:
                        top_logprobs_dicts = top_logprobs_dicts
                    else:
                        top_logprobs_dicts = top_logprobs_dicts[1:]
                res_d.append({"text": result, "tokens": convert_tokens(result, tokenizer), "logprobs": logprobs, "top_logprobs_dicts": top_logprobs_dicts})
            completions.append(res_d)
        print(completions)
        return {"completions": completions, "input_length": len(data), "request_time": end_time - start_time, "request_datetime": start_time}  # 返回布尔值
    except Exception as e:
        print("Exception", e)
        return {"completions": [], "input_length": 0, "request_time": 0, "request_datetime": start_time}  # 返回布尔值


def initialize(model_name):
    # print(os.listdir("./models"))
    print("current path")
    """load module, executed once at the start of the service
    do service intialization and load models in this function.
    """
    global model, tokenizer, device
    print(f"model_name=={model_name}")
    # checkpoint = os.path.join(model_name, "gogpt-7b/")
    checkpoint = model_name
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print("tokenzier load succes")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    print("model load succes")
    model.half()
    print("half 模式")
    device = "cuda"  # for GPU usage or "cpu" for CPU usage
    model = model.to(device)
    model.eval()
    return


def process(inputs):
    print("==================================")
    inp = json.loads(inputs)
    print(inp)
    text = inp["text"]
    print(text)
    print("==================================")
    res = predict(text)["completions"]
    print(res)
    if res:
        return {"history": res[0][0]["text"]}
    return res


if __name__ == '__main__':
    initialize("/home/searchgpt/pretrained_models/gogpt-7b-v2")
    predict("如何学习数学?")