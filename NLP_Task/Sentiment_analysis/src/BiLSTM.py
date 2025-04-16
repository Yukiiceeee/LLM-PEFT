from transformers import AutoTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset
from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from torch.nn import functional as F
import argparse
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import csv
import sys
import json
from dataset import SentimentDataset
from dataset import load_datasets
import re
import os

class BiLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_layers, num_classes, dropout=0.2, freeze_bert=False):
        super(BiLSTM, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state  # [batch_size, seq_len, 768]
        
        lengths = attention_mask.sum(dim=1)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            bert_output, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        attention_weights = self.attention(output)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        text_representation = torch.sum(attention_weights * output, dim=1)
        
        logits = self.fc(self.dropout(text_representation))
        
        return logits

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_val_acc = 0.0
    training_states = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'learning_rates': []
    }

    training_states['learning_rates'].append(optimizer.param_groups[0]['lr'])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            lengths = attention_mask.sum(dim=1)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            current_loss = train_loss / (batch_idx + 1)
            current_acc = 100. * train_correct / train_total
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc='Validating')
            for batch_idx, batch in enumerate(val_progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                lengths = attention_mask.sum(dim=1)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                current_val_loss = val_loss / (batch_idx + 1)
                current_val_acc = 100. * val_correct / val_total
                val_progress_bar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        training_states['learning_rates'].append(current_lr)

        training_states['epochs'].append(epoch + 1)
        training_states['train_loss'].append(train_loss)
        training_states['train_acc'].append(train_acc)
        training_states['val_loss'].append(val_loss)
        training_states['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            training_states['best_val_acc'] = best_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved to {os.path.join(save_dir, "best_model.pth")}')

    with open(os.path.join(save_dir, 'train_state.json'), 'w', encoding='utf-8') as f:
        json.dump(training_states, f, indent=4)
    
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    return best_val_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            lengths = attention_mask.sum(dim=1)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_loss = test_loss/len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/data')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/models/BiLSTM')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    model = BiLSTM(
        bert_model_name=args.model_path,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 分层学习率
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate * 10}  # BiLSTM部分使用更大的学习率
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2,
        verbose=True
    )

    # Data Prepare
    train_loader, val_loader, test_loader = load_datasets(
        data_dir=args.data_dir,
        tokenizer_name=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    best_val_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir
    )

    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc
    }
    
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)
    
    print(f'\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()