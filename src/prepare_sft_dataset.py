#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: prepare_sft_dataset.py
@time: 2023/08/02
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

import datasets
import pandas as pd

dataset = datasets.load_dataset('/data/searchgpt/data/instruction_data/instruction_merge_set/data')
df = dataset['train'].to_pandas()
print(df['id'].apply(lambda x: x.split('_')[0]).value_counts()) # 46
"""
train                                       2015045
firefly-train-1.1M                          1645916
Belle                                       1429442
data                                         886545
multiturn                                    831015
ultrachat                                    682847
Belle.train                                  540702
generated                                    396001
guanaco                                      251518
school                                       248300
HealthCareMagic-200k                         207407
alpaca                                       204164
comparision                                  103989
instinwild                                   103607
general                                       93043
sg                                            76838
stackoverflow                                 57046
quora                                         54454
medical                                       47107
final                                         44206
all                                           36711
paper                                         29012
gpt4-instruct-dedupe-only-dataset             18088
unified                                       15815
icliniq-15k                                   15134
databricks-dolly-15k                          14974
chatalpaca                                     9997
toolformer-dedupe-only-dataset                 7655
GenMedGPT-5k                                   5452
roleplay-simple-deduped-roleplay-dataset       3146
reasoning                                       800
hotels                                          400
restaurants                                     245
event                                           240
scientific                                      100
home                                             80
car                                              48
pet                                              48
tech                                             32
flight                                           32
legal                                            16
insurance                                        16
real                                             16
travel                                           16
fitness                                          16
job                                              16

"""
df['id_prefix'] = df['id'].apply(lambda x: x.split('_')[0])
df[df['id_prefix']=='icliniq-15k']

tmp=[]

for cate in df['id_prefix'].unique():
    tmp.append(df[df['id_prefix']==cate][:20000])
sample_df=pd.concat(tmp,axis=0)
print(sample_df['id_prefix'].value_counts())



sample_df['con_lens']=sample_df['conversations'].apply( lambda  x:len(x))
print(sample_df['con_lens'].describe())
print(sample_df['con_lens'].value_counts())
print(sample_df[sample_df['con_lens']==0])
print("sample_df.shape",sample_df.shape)
sample_df=sample_df[sample_df['con_lens']!=0].reset_index(drop=True)

sample_df['conversations_len']=sample_df['conversations'].map(len)
sample_df=sample_df[sample_df['conversations_len']<=10].reset_index(drop=True)
print("sample_df.shape",sample_df.shape)

sample_df.to_parquet('sample_df_532k.parquet')
sample_df.to_json("sample_df_532k.json", orient="records", lines=True, force_ascii=False)
