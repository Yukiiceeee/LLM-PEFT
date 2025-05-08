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

class SentimentDataset_lora(Dataset):
    system_prompt = "你是一个情感分析助手。请分析以下评论的情感倾向。输出应该是0（消极）或1（积极）。"
    
    input_template = (
        "系统: {system}\n\n"
        "人类: 请分析这段评论的情感倾向：\n{review}\n\n"
        "助手: "
    )
    
    output_template = "这段评论的情感倾向是{label}（{sentiment}）。"
    
    sentiment_map = {
        0: "消极",
        1: "积极"
    }

    def __init__(self, file_path, tokenizer_name="bert-base-chinese", max_length=256):
        self.data = pd.read_csv(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def get_input_connected_with_no_label(self, idx):
        text = str(self.data.iloc[idx]['review'])
        input_text = self.input_template.format(
            system=self.system_prompt,
            review=text
        )
        
        tokens = self.tokenizer(
            input_text,
            padding_side="left",
            return_tensors="pt",
            return_token_type_ids=False
        )

        return tokens

    def get_input_connected_with_label(self, idx):
        text = str(self.data.iloc[idx]['review'])
        label = int(self.data.iloc[idx]['label'])
        
        input_text = self.input_template.format(
            system=self.system_prompt,
            review=text
        )
        output_text = self.output_template.format(
            label=label,
            sentiment=self.sentiment_map[label]
        )
        
        full_text = input_text + output_text
        
        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            padding_side='left',
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )
        
        output_tokens = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding='max_length',
            padding_side='left',
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False
        )
        
        tokens["input_ids"] = tokens["input_ids"].flatten()
        tokens["attention_mask"] = tokens["attention_mask"].flatten()
        tokens["labels"] = output_tokens["input_ids"].flatten()
        tokens["labels"][tokens["labels"] == self.tokenizer.pad_token_id] = -100

        return tokens

    def __getitem__(self, idx):
        return self.get_input_connected_with_label(idx)

def load_datasets_lora(data_dir, tokenizer_name="bert-base-chinese", max_length=256):
    train_dataset = SentimentDataset_lora(
        os.path.join(data_dir, 'train_data.csv'),
        tokenizer_name,
        max_length
    )
    val_dataset = SentimentDataset_lora(
        os.path.join(data_dir, 'val_data.csv'),
        tokenizer_name,
        max_length
    )
    test_dataset = SentimentDataset_lora(
        os.path.join(data_dir, 'test_data.csv'),
        tokenizer_name,
        max_length
    )

    return train_dataset, val_dataset, test_dataset

def main():
    train_dataset, val_dataset, test_dataset = load_datasets_lora("/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/data")
    
    tokenizer_name="bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("输入：",tokenizer.decode(train_dataset[0]['input_ids']))
    print("标签：",tokenizer.decode(train_dataset[0]['labels'][train_dataset[0]['labels'] != -100])) 

    print("\n=== 张量信息 ===")
    print(f"Input shape: {train_dataset[0]['input_ids'].shape}")
    print(f"Attention mask shape: {train_dataset[0]['attention_mask'].shape}")
    print(f"Labels shape: {train_dataset[0]['labels'].shape}")
    print(f"Input: {train_dataset[0]['input_ids']}")
    print(f"Attention mask: {train_dataset[0]['attention_mask']}")
    print(f"Labels: {train_dataset[0]['labels']}")

    
if __name__ == "__main__":
    main()
        
        
        

