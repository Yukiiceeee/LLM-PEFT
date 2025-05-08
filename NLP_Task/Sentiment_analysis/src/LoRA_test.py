from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, Optional, Sequence, List
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as LORA_TARGET_MAP
import json
import torch
import os
import copy
import logging
import transformers
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from tqdm import tqdm
from dataset import SentimentDataset
from dataset import load_datasets
from dataset import load_datasets_lora
from dataset import SentimentDataset_lora

def extract_sentiment(response, prompt):
    result = response.replace(prompt, "").strip()
    
    if "积极" in result:
        return 1
    elif "消极" in result:
        return 0
    elif "正面" in result:
        return 1
    elif "负面" in result:
        return 0
    elif "1" in result:
        return 1
    elif "0" in result:
        return 0
    else:
        return 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model_path = "/d2/mxy/Models/Qwen2-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"加载Base模型: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model = base_model
    
    # lora_model_path = "/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/models/LoRA"
    # print(f"加载LoRA模型: {lora_model_path}")
    # model = PeftModel.from_pretrained(
    #     base_model,
    #     lora_model_path,
    #     torch_dtype=torch.float16
    # )
    # print("LoRA模型加载完成")

    model.eval()

    test_data_path = "/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/data/test_data.csv"
    df = pd.read_csv(test_data_path)
    
    print(f"加载了 {len(df)} 条测试数据")
    
    predictions = []
    ground_truth = df['label'].tolist()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理测试数据"):
        text = row['review']
        label = row['label']
        
        prompt = f"""你是一个情感分析助手。请分析以下评论的情感倾向。输出应该是0（消极）或1（积极）。
                     请分析这段评论的情感倾向： "{text}"
                     情感分类结果:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        pred_label = extract_sentiment(response, prompt)
        predictions.append(pred_label)
        
        # print(f"\n示例 {idx+1}:")
        # # print(f"文本: {text}")
        # print(f"真实标签: {label} ({'积极' if label == 1 else '消极'})")
        # print(f"预测标签: {pred_label} ({'积极' if pred_label == 1 else '消极'})")
        # # print(f"模型回答: {response}")
    
    accuracy = accuracy_score(ground_truth, predictions)
    
    print("\n模型评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    
    results_df = pd.DataFrame({
        'review': df['review'],
        'true_label': ground_truth,
        'predicted_label': predictions
    })
    results_df['accuracy'] = accuracy
    
    results_path = "/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/models/Base_Model/prediction_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    print(f"\n预测结果已保存至: {results_path}")

if __name__ == "__main__":
    main()

