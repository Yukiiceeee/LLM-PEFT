from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import sys
import json
import re
import os

class SentimentDataset(Dataset):
    def __init__(self, file_path, tokenizer_name="bert-base-chinese", max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['review'])
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            padding_side='left',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_datasets(data_dir, tokenizer_name="bert-base-chinese", max_length=512, batch_size=16):
    train_dataset = SentimentDataset(
        os.path.join(data_dir, 'train_data.csv'),
        tokenizer_name,
        max_length
    )
    val_dataset = SentimentDataset(
        os.path.join(data_dir, 'val_data.csv'),
        tokenizer_name,
        max_length
    )
    test_dataset = SentimentDataset(
        os.path.join(data_dir, 'test_data.csv'),
        tokenizer_name,
        max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

