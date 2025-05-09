import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import jieba
from collections import Counter
import numpy as np
import os
import sys
from transformers import BertTokenizer, BertModel
sys.path.append('/d2/mxy/LLM-PEFT')
from Transformer.transformer import Transformer

class BertTransformer(nn.Module):
    def __init__(self, transformer_model, bert_model):
        super(BertTransformer, self).__init__()
        self.bert = bert_model
        self.transformer = transformer_model
        self.projection = nn.Linear(768, transformer_model.encoder.d_model)
        
    def forward(self, src, tgt):
        bert_output = self.bert(src)[0]
        projected_encoding = self.projection(bert_output)
        return self.transformer(projected_encoding, tgt)

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_src_len=512, max_tgt_len=50):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.build_vocab()
        
    def build_vocab(self):
        word_counts = Counter()
        for item in self.data:
            word_counts.update(jieba.lcut(item['title']))  # 只需要为解码器构建词表
            
        self.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        for word, _ in word_counts.most_common(30000):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def encode_source(self, text):
        encoded = self.tokenizer(text, 
                               max_length=self.max_src_len,
                               padding='max_length',
                               truncation=True,
                               return_tensors='pt')
        return encoded['input_ids'].squeeze(0)
    
    def encode_target(self, text, max_len):
        tokens = jieba.lcut(text)
        ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        ids = ids[:max_len-2]
        ids = [self.word2idx['<BOS>']] + ids + [self.word2idx['<EOS>']]
        ids = ids + [self.word2idx['<PAD>']] * (max_len - len(ids))
        return ids
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids = self.encode_source(item['content'])
        tgt_ids = self.encode_target(item['title'], self.max_tgt_len)
        return {
            'src': src_ids,
            'tgt': torch.tensor(tgt_ids)
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/d2/mxy/LLM-PEFT/NLP_Task/Summary_generation/data/lcsts_data.json'
    batch_size = 16
    epochs = 10
    d_model = 768
    hidden_dim = 3072
    num_heads = 12
    num_layers = 12
    max_src_len = 512
    max_tgt_len = 50
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    
    for param in bert_model.parameters():
        param.requires_grad = False
    
    dataset = TextDataset(data_path, bert_tokenizer, max_src_len, max_tgt_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    transformer = Transformer(
        src_pad_idx=bert_tokenizer.pad_token_id,
        tgt_pad_idx=dataset.word2idx['<PAD>'],
        src_voc_size=bert_tokenizer.vocab_size,
        tgt_voc_size=dataset.vocab_size,
        max_len=max(max_src_len, max_tgt_len),
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device
    )
    
    model = BertTransformer(transformer, bert_model).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    
    warmup_steps = 4000
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    )
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Epoch: {epoch+1}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
            
if __name__ == '__main__':
    main()
