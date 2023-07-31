## 模型训练问题记录

### 1 在gogpt底座微调（sft）阶段，出现学习率为0
学习率慢慢变为0，一开始还是正常的学习预热+衰减，但是到后期（比如迭代2000步）之后，学习率直接变为0了。
```text
gradient_accumulation_steps设置为1

在Github Issues看到也和deepspeed的学习率规划器设置冲突了，去除deepspeed中关于优化器和学习率规划器的设置即可
```
![issue1.png](assets%2Fissue1.png)

### 2 在gogpt底座微调（sft）阶段，出现loss为0
![img.png](assets/issue2.png)

### 3 指令微调的时候模板应该怎么设计？不用模板可以吗？

常见的是stanford_alpaca中模板：
```text
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
```


Llama2中的模板
```text
instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

```

Linly-AI中模板
```text
### Instruction:{prompt.strip()}  ### Response:
```

NousResearch

```text
### Instruction:
<prompt>

### Response:
<leave a newline blank for model to respond>
```

```text
### Instruction:
<prompt>

### Input:
<additional context>

### Response:
<leave a newline blank for model to respond>

```

Yayi
> https://huggingface.co/wenge-research/yayi-7b-llama2
> 
```text
prompt = "你是谁？"
formatted_prompt = f"""<|System|>:
You are a helpful, respectful and honest assistant named YaYi developed by Beijing Wenge Technology Co.,Ltd. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

<|Human|>:
{prompt}

<|YaYi|>:
"""
```

StableBeluga2
> https://huggingface.co/stabilityai/StableBeluga2
```text
### System:
This is a system prompt, please behave and help the user.

### User:
Your prompt here

### Assistant:
The output of Stable Beluga 2
```
比如
```text
system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

message = "Write me a poem please"
prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
```


llama-2-70b-Guanaco-QLoRA-fp16
```text
### Human: {prompt}
### Assistant:
```

```text
prompt = "Introduce yourself"
formatted_prompt = (
    f"A chat between a curious human and an artificial intelligence assistant."
    f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    f"### Human: {prompt} ### Assistant:"
)
```

### 4 多轮对话数据怎么构造
- 方式1：训练时，我们将多轮对话拼接成如下格式，然后进行tokenize。其中<s>表示bos_token，</s> 表示eos_token。
```text
<s>input1</s>target1</s>input2</s>target2</s>...
```
> https://github.com/yangjianxin1/Firefly

在计算loss时，我们通过mask的方式，input部分的loss不参与参数更新，只有“target”部分的loss参与参数更新。 这种方式充分利用了模型并行计算的优势，训练更加高效，且多轮对话中的每个target部分都参与了训练，训练更充分。 否则，就需要把一个n轮对话，拆分成n条数据，且只计算最后一个target的loss，大大降低了训练效率。


### 5 Bugs:FAILED: multi_tensor_adam.cuda.o 

```text
原因：CUDA环境变量没有配置正确：
/bin/sh: /usr/usr/local/cuda-12.2/bin/nvcc: No such file or directory
```
改成`/usr/local/cuda-12.2/bin/nvcc`即可

正确配置如下
```text
export PATH=/usr/local/cuda-12.2/bin:$PATH  
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2/
```

### 6 常见英文微调数据集有哪些
```text
1、 模型1 https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b


GPTeacher was made available by Teknium
https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct

Wizard LM by nlpxucan
https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k


Nous Research Instruct Dataset was provided by Karan4D and HueminArt.




GPT4-LLM and Unnatural Instructions were provided by Microsoft
https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned?clone=true

Airoboros dataset by jondurbin
https://huggingface.co/datasets/jondurbin/airoboros-gpt4-m2.0?clone=true


Camel-AI's domain expert datasets are from Camel-AI
CodeAlpaca dataset by Sahil 2801.



2、模型2 https://huggingface.co/stabilityai/StableBeluga2


/home/searchgpt/pretrained_models/data/WizardLM_evol_instruct_V2_143k/WizardLM_evol_instruct_V2_143k.jsonl


/home/searchgpt/pretrained_models/gogpt2-7b

/home/searchgpt/yq/Firefly/output/firefly-llama2-7b/final


/home/searchgpt/yq/Firefly/checkpoint/gogpt2-7b-qlora-sft-merge

Run "huggingface-cli lfs-enable-largefiles ./path/to/your/repo" and try again.
Run "huggingface-cli lfs-enable-largefiles ./path/to/your/repo" and try again.


/data/searchgpt/yq/GoGPT/outputs-pt-v1-13b-llama2
```