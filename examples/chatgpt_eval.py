other_prompt="Based on the output, your task is to determine the better instruction. The evaluation criteria for the better instruction include aligning with the output content and maximizing precision while avoiding ambiguity. Only give the better one in the form {'output': 'instruction x is better'} without additional explanation."
import jsonlines
import openai
openai.api_key = ''
def normal_result(s):
    if s==None:
        return 'error'
    s = " ".join(s.lower().split())
    if "instruction a" in s:
        return "A"
    elif "better instruction is a" in s:
        return "A"
    elif "a is better" in s:
        return "A"
    elif "a is the better" in s:
        return "A"
    else:
        return "B"
def get_answer(prompt):
    completion_turbo_with_api = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    ans = completion_turbo_with_api["choices"][0]["message"]["content"]
    return ans
import time
from tqdm import tqdm
chatgpt_result=[]
better=0
same=0
worse=0
for i in range(len(data1)):
    assert data1[i]['input']==data2[i]['input']
    query = data1[i]['input']
    ans1 = data1[i]['model_output']
    ans2 = data2[i]['model_output']
    prompt = "{}\noutput: {}\nOptions:\ninstruction A: '{}'\ninstruction B: '{}'".format(other_prompt, query, ans1, ans2)
    ans = get_answer(prompt)
    print(ans)
    res = normal_result(ans)
    time.sleep(2)
    rev_prompt = "{}\noutput: {}\nOptions:\ninstruction A: '{}'\ninstruction B: '{}'".format(other_prompt, query, ans2, ans1)
    rev_ans = get_answer(rev_prompt)
    print(rev_ans)
    rev_res = normal_result(rev_ans)
    if res == 'error' or rev_res=='error':
        print('error')
    elif res == rev_res:
        same+=1
    elif res == 'A':
        better+=1
    elif res=='B':
        worse+=1
    print(better,same,worse)
    time.sleep(2)