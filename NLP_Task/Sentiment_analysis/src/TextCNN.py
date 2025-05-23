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
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class TextCNN(nn.Module):
    def __init__(self, bert_model_name, num_channels, kernel_sizes, num_classes, dropout=0.2, freeze_bert=False):
        super(TextCNN, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        d_model = 768

        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=num_channels,
                kernel_size=ks
            )
            for ks in kernel_sizes
        ])

        self.relu = nn.ReLU()

        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # bert_output.shape: [batch_size, seq_len, 768]
        bert_output = outputs.last_hidden_state
        bert_output = bert_output.permute(0, 2, 1)

        conved = [self.relu(conv(bert_output)) for conv in self.convs]
        conved_pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]
        conved_pooled = [conv.squeeze(2) for conv in conved_pooled]
        # conved_cat.shape: [batch_size, num_channels * len(kernel_sizes)]
        conved_cat = self.dropout(torch.cat(conved_pooled, dim=1))

        logits = self.fc(conved_cat)

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
    
    step_states = {
        'steps': [],
        'train_loss': []
    }
    global_step = 0

    training_states['learning_rates'].append(optimizer.param_groups[0]['lr'])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        all_labels = []
        all_predictions = []

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
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            global_step += 1
            if global_step % 10 == 0:
                step_states['steps'].append(global_step)
                step_states['train_loss'].append(loss.item())
                
                plt.figure(figsize=(10, 6))
                plt.plot(step_states['steps'], step_states['train_loss'])
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training Loss per 10 Steps')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, 'step_loss_curve.png'))
                plt.close()

            current_loss = train_loss / (batch_idx + 1)
            current_acc = 100. * train_correct / train_total
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels_list = []
        
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
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

                current_val_loss = val_loss / (batch_idx + 1)
                current_val_acc = 100. * val_correct / val_total
                val_progress_bar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels_list, val_predictions, average='binary'
        )

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
        print(f'Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'lr': current_lr
        }
        
        with open(os.path.join(save_dir, f'epoch_{epoch+1}_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(epoch_metrics, f, indent=4)
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(training_states['epochs'], training_states['train_loss'], label='Train Loss')
        plt.plot(training_states['epochs'], training_states['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss per Epoch')
        
        plt.subplot(2, 2, 2)
        plt.plot(training_states['epochs'], training_states['train_acc'], label='Train Acc')
        plt.plot(training_states['epochs'], training_states['val_acc'], label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy per Epoch')
        
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            training_states['best_val_acc'] = best_val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved to {os.path.join(save_dir, "best_model.pth")}')

    with open(os.path.join(save_dir, 'train_state.json'), 'w', encoding='utf-8') as f:
        json.dump(training_states, f, indent=4)
    
    with open(os.path.join(save_dir, 'step_loss.json'), 'w', encoding='utf-8') as f:
        json.dump(step_states, f, indent=4)
    
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    return best_val_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_predictions = []
    all_labels = []
    
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
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss/len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    metrics = {
        'accuracy': test_acc,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'loss': test_loss
    }
    
    return test_loss, test_acc, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/data')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='/d2/mxy/LLM-PEFT/NLP_Task/Sentiment_analysis/models/TextCNN')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    kernel_sizes = [2, 3, 4]

    model = TextCNN(
        bert_model_name=args.model_path,
        num_channels=100,
        kernel_sizes=kernel_sizes,
        num_classes=2,
        dropout=0.2,
        freeze_bert=False
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
         'lr': args.learning_rate * 10}
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
    test_loss, test_acc, test_metrics = evaluate(model, test_loader, criterion, device)
    
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'best_val_accuracy': best_val_acc
    }
    
    with open(os.path.join(args.save_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)
    
    print(f'\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f'Precision: {test_metrics["precision"]:.2f}%')
    print(f'Recall: {test_metrics["recall"]:.2f}%')
    print(f'F1 Score: {test_metrics["f1"]:.2f}%')

if __name__ == "__main__":
    main()