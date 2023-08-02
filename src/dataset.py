import json

from loguru import logger
from torch.utils.data import Dataset

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
dummy_message = {
    "system": """\
            You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    "id": "dummy_message",
    "conversations": [
        {"from": "human", "value": "Who are you?"},
        {"from": "gpt", "value": "I am your virtual friend."},
        {"from": "human", "value": "What can you do?"},
        {"from": "gpt", "value": "I can chat with you."}
    ]
}


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # https://github.com/LinkSoul-AI/Chinese-Llama-2-7b/blob/main/train.py
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversations']

        if "instruction" in data and len(data["instruction"]) > 0:
            system = data["instruction"]
        else:
            system = dummy_message["system"]
        system = B_SYS + system + E_SYS
        # add system before the first content in conversations
        data["conversations"][0]['value'] = system + data["conversations"][0]['value']

        # 收集多轮对话
        utterances = []
        for idx, x in enumerate(conversation):
            # if x['from'] == 'human' and idx % 2 == 0:
            if x['from'] == 'human':
                content = x['value'].replace('MOSS', 'GoGPT').replace('moss', 'GoGPT')
                content = content.strip()
                content = f"{B_INST} {content} {E_INST} "
                utterances.append(content)
            # if x['from']=='gpt'and idx%2==1:
            else:
                # assert role == "gpt"
                content = x['value'].replace('MOSS', 'GoGPT').replace('moss', 'GoGPT')
                content = f"{content} "
                utterances.append(content)
        print(utterances)
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]
        target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            input_ids += (utterances_id + [self.eos_token_id])
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)
        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs
