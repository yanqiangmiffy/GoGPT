from datasets import load_dataset
raw_datasets = load_dataset(
        'json',
        data_files='data/finetune/alpaca/alpaca_data.json',
    )
print(raw_datasets)