# GoGPT

> GoGPT:基于Llama/Llama 2训练的中英文增强大模型

<p align="center">
    <br>
    <img src="resources/assets/gogpt-banner-tou.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
</p>

> ICT中英文底座增强大模型：70亿参数、130亿参数

## step1：训练分词器

[🐱怎么从零到一训练一个LLM分词器](https://github.com/yanqiangmiffy/how-to-train-tokenizer)

```text
├── data
│     └── corpus.txt 训练语料
├── llama
│     ├── tokenizer_checklist.chk
│     └── tokenizer.model
├── merged_tokenizer_hf 合并结果 hf格式
│     ├── special_tokens_map.json
│     ├── tokenizer_config.json
│     └── tokenizer.model
├── merged_tokenizer_sp
│     └── open_llama.model # 
├── merge_tokenizer
│     └── tokenizer.model
├── open_llama.model 训练的sp模型
├── open_llama.vocab 训练的sp词汇表
├── README.md
├── step0_step0_process_text.py 基于多分数据集准备训练语料
├── step1_make_corpus.py 基于中文Wikipedia数据准备训练语料
├── step2_train_tokenzier.py  训练分词器
├── step3_tokenzier_segment.py 测试训练后的模型，包括编码和解码测试样例
└── step4_merge_tokenizers.py 与原版llama的分词器进行合并，得到hf格式的tokenizer

```

## step2：二次预训练

> 在中文预训练语料上对LLaMA进行增量预训练、继续预训练

## step3: 有监督微调

- belle数据：120k数据  v1
- stanford_alapca：52k数据 v2
- [sharegpt](data%2Ffinetune%2Fsharegpt):90k数据

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。

模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。

对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

