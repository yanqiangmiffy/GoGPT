#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: load_sft_dataset_example.py
@time: 2023/07/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import logging
import os
from typing import List
from typing import Optional

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []
    # all_file_size = []

    for (root, dir, file_name) in os.walk(dir_name):
        for temp_file in file_name:
            standard_path = f"{root}/{temp_file}"

            all_file_list.append(standard_path)

    return all_file_list


def load_dataset_from_path(data_path: Optional[str] = None,
                           cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    print(all_file_list)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info("load files %d number", len(all_file_list))

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets

#
# dataset = load_dataset_from_path(
#         data_path='../data/finetune/examples',
#     )
#
# print(dataset)
# print(dataset[0])


dataset = load_dataset_from_path(
        data_path='../data/finetune/opendata',
    )

print(dataset)
print(dataset[0])