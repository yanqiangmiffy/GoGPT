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

### 4 多轮对话数据怎么构造
- 方式1：训练时，我们将多轮对话拼接成如下格式，然后进行tokenize。其中<s>表示bos_token，</s> 表示eos_token。
```text
<s>input1</s>target1</s>input2</s>target2</s>...
```
> https://github.com/yangjianxin1/Firefly

在计算loss时，我们通过mask的方式，input部分的loss不参与参数更新，只有“target”部分的loss参与参数更新。 这种方式充分利用了模型并行计算的优势，训练更加高效，且多轮对话中的每个target部分都参与了训练，训练更充分。 否则，就需要把一个n轮对话，拆分成n条数据，且只计算最后一个target的loss，大大降低了训练效率。