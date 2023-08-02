# GoGPT

> GoGPT:ICT中英文底座增强大模型,基于Llama/Llama 2训练的底座大模型,参数规模包括70亿参数、130亿参数

<p align="center">
    <br>
    <img src="resources/assets/gogpt-banner-tou.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
</p>

## 更新
- 8月1号  gogpt2-13b 模型开源，基于gogpt2-13b-pretrain在120多万指令数据进行sft 🔥🔥
- 7月31号 gogpt2-13b-pretrain 模型开源 ，基于中文语料进行中文词表扩充以及继续与继续预训练，词表大小68420🔥🔥
- 7月28号 gogpt2-7b-pretrain 模型开源，基于中文语料进行中文词表扩充以及继续与继续预训练，词表大小68420 🔥🔥
- 7月22号 gogpt2-7b 模型开源 ，基于gogpt2-7b-pretrain在120多万指令数据进行sft🔥🔥

## ⏬模型下载

### 🤗HuggingFace仓库
Huggingface上提供了GoGPT权重，目前开放了gogpt-7b、gogpt2-7b、gogpt-13b权重

| 模型名称                                                        | 基座模型      | 模型大小 | 下载地址                                            |
|-------------------------------------------------------------|-----------|------|-------------------------------------------------|
| [golaxy/gogpt-7b](https://huggingface.co/golaxy/gogpt-7b)   | Llama-7b  | 7B   | [模型下载](https://huggingface.co/golaxy/gogpt-7b)  |
| [golaxy/gogpt2-7b](https://huggingface.co/golaxy/gogpt2-7b) | Llama2-7b | 7B   | [模型下载](https://huggingface.co/golaxy/gogpt2-7b) |     
| [golaxy/gogpt2-7b-pretrain](https://huggingface.co/golaxy/gogpt2-7b-pretrain) | Llama2-7b | 7B   | [模型下载](https://huggingface.co/golaxy/gogpt2-7b-pretrain) |    
| [golaxy/gogpt2-13b-pretrain](https://huggingface.co/golaxy/gogpt2-13b-pretrain) | Llama2-13b | 13B   | [模型下载](https://huggingface.co/golaxy/gogpt2-13b-pretrain) |    
| [golaxy/gogpt2-13b](https://huggingface.co/golaxy/gogpt2-13b) | Llama2-13b | 13B   | [模型下载](https://huggingface.co/golaxy/gogpt2-13b) |   
### 🚀代码调用
- golaxy/gogpt-7b
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("golaxy/gogpt-7b")
model = AutoModelForCausalLM.from_pretrained("golaxy/gogpt-7b")
```
- golaxy/gogpt2-7b
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("golaxy/gogpt2-7b")
model = AutoModelForCausalLM.from_pretrained("golaxy/gogpt2-7b")
```

### CLI测试

```shell
python cli.py
```

### Gradio部署

```shell
python app.py
```

![image](https://github.com/yanqiangmiffy/GoGPT/assets/15925090/c295924d-a353-40d7-9d1a-a268026dc898)

## 大模型技术交流群

![ce6ad8231b096c380b6f575c7095a4c](https://github.com/yanqiangmiffy/GoGPT/assets/15925090/5403509e-26ee-44d8-9ab5-336959cd9e7c)



## 测试效果
- 西瓜和南瓜有什么区别
```
1. 外观：西瓜通常是圆形或椭圆形的，有浅黄色或棕色的外皮，内部有红色或黄色的果肉。而南瓜则是一个圆形或椭圆形的，通常有浅黄色或棕色的外皮，内部有橙黄色的果肉。
2. 品种：西瓜主要分为三种品种：小型、中等和巨型。小型西瓜通常只有一个或两个果核，中等西瓜通常有两个或三个果核，巨型西瓜则有四个或更多果核。南瓜主要分为两种品种：南 瓜和萝卜南瓜。南瓜通常有两个或三个果核，萝卜南瓜则有四个或更多果核。
3. 生长环境：西瓜需要温暖的气候，并且需要充足的阳光。它们通常生长在热带或亚热带地区。南瓜则在温带和寒带地区生长。南瓜需要更长的冬季来生长，而西瓜通常在春季和夏季生 长。
4. 营养价值：西瓜和南瓜都是富含维生素C和纤维素的水果。西瓜还富含抗氧化剂和钾，而南瓜富含叶酸和维生素E。
总的来说，西瓜和南瓜是两种完全不同的果类，它们在外观、品种、生长环境和营养价值上有很多区别。
```

- 如何提高我的时间管理技能？
```
Response:  1. 制定计划和目标：制定每周或每日的计划和目标，以帮助您更好地分配时间。
2. 优先考虑最重要的任务：优先完成最重要的任务，这有助于避免分心和浪费时间。
3. 避免拖延：尝试避免拖延，尽早开始任务，这样您可以更容易地完成它们。
4. 集中注意力：集中注意力，避免同时处理多个任务，这有助于减少分心和提高效率。5. 利用工具和技术：利用时间管理工具和技术，如日历、提醒和待办事项列表，以帮助您更好地管理时间。
6. 学会说“不”：学会拒绝不必要的事情，这有助于您更好地利用时间，并专注于最重要的任务。
7. 保持积极心态：保持积极心态，相信自己能够管理时间，并尝试不断学习和发展自己的时间管理技能。
8. 休息和放松：尝试保持工作和生活的平衡，并定期休息和放松，这有助于提高时间管理技能和提高工作效率。
```

- Python和JavaScript编程语言的主要区别是什么？
```
1. 语法不同：Python是一种面向对象的编程语言，它的语法比较严格，需要使用花括号来标识代码块。JavaScript是一种脚本语言，它的语法相对比较松散，需要使用花括号或缩进来标 识代码块。
2. 应用领域不同：Python广泛应用于数据分析、机器学习、人工智能等领域，由于其可读性好、语法清晰、功能强大等特点。JavaScript则主要应用于前端开发、后端开发、移动端开发 等领域。
3. 运行环境不同：Python的运行环境较为独立，可以在不同的操作系统上运行。JavaScript的运行环境相对较复杂，需要考虑浏览器和Web服务器的兼容性问题。
4. 应用场景不同：Python更适合进行科学计算、数据分析、机器学习等方面的应用，如Pandas、Numpy等库；而JavaScript则更适合进行前端开发、后端开发、移动端开发等领域的脚本编程，如React、Angular等框架。
5. 库和框架不同：Python拥有丰富的标准库和第三方库，如Django、Flask等；而JavaScript则拥有流行的Web开发框架，如React、Angular等。
```

- AI会如何影响农业生产？
```
AI将有助于提高农业生产效率，优化作物种植和收成过程，以及提高农产品的质量和品质。
AI技术可以应用于植物监测、灌溉管理、病虫害防治、土壤分析等方面，减少人力 成本，提高生产效率，同时也可以降低农业生产中的资源浪费和环境污染。例如，AI可以监测土壤湿度、气象状况等环境因素，帮助农民决定何时种植、灌溉和施肥，从而提高农作物的产量和质量。
另外，AI还可以通过监测植物健康状况，及时发现和预防病虫害的发生，提高农作物的产量和质量。此外，AI还可以帮助农民预测未来天气、干旱和洪涝灾害，制定更有效的灾害应对计划。
总之，AI有望成为农业生产中不可或缺的技术，在未来为农业生产带来更大的改进和效益。
```
- 翻译成英文：过去都是假的，回忆是一条没有归途的路，以往的一切春天都无法复原，即使最狂热最坚贞的爱情，归根结底也不过是一种瞬息即逝的现实，唯有孤独永恒。
```
Response:  Translation into English: 
The past is false, memories are a dead-end road, all previous springs are hopelessly incapable of being restored, even the most passionate and steadfast love is ultimately a fleeting reality,
and only solitude is eternal
```

- 请用python编程语言实现快速排序算法的代码
```
快速排序（Quick Sort）是一种常用的排序算法，基本思想是通过一次排序将数组分为两个部分，其中一个部分的所有元素都比另一个小，然后再分别对这两个部分进行排序 ，直到整个数组有序。
以下是使用Python语言实现快速排序的代码：
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```
在这个实现中，首先判断数组的长度是否小于等于1，如果是，则直接返回该数组。否则，选择一个枢轴（pivot）元素，将数组分为三个部分：小于枢轴的左部分、等于枢轴的中间部分、大于枢轴的右部分。接着递归地对左部分和右部分进行排序，并将它们与中间部分连接起来。最后返回排好序的数组。
以下是使用上述代码对数组进行排序的示例：
```python
arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_arr = quick_sort(arr)
print(sorted_arr)
```
```
输出结果为：[1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```

## 训练细节

### step1：训练分词器

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

### step2：二次预训练

> 在中文预训练语料上对LLaMA进行增量预训练、继续预训练

### step3: 有监督微调

- belle数据：120k数据 v1
- stanford_alapca：52k数据 v2
- [sharegpt](data%2Ffinetune%2Fsharegpt):90k数据
  
![image](https://github.com/yanqiangmiffy/GoGPT/assets/15925090/64125339-b6d1-42ba-a271-2e9f90c161ee)

### step4: 强化学习

> TODO

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。

模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。

对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。


## 研究与开发团队

本项目由网络数据科学与技术重点实验室GoGPT团队完成，团队指导老师为郭嘉丰研究员。
