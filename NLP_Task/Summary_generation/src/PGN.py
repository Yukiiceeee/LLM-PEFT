import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import numpy as np
from typing import List, Tuple, Dict
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Encoder(nn.Module):
    def __init__(self, bert_model: BertModel, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(
            768,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():  # 冻结BERT参数
            bert_outputs = self.bert(x, attention_mask=attention_mask)
            embedded = bert_outputs.last_hidden_state
        
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        
    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        
        attention = attention.masked_fill(mask == 0, float('-inf'))
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, bert_model: BertModel, hidden_size: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = bert_model.config.vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = bert_model.embeddings.word_embeddings
        self.attention = Attention(hidden_size)
        
        self.lstm = nn.LSTM(
            hidden_size * 2 + 768,  # 注意力上下文 + BERT嵌入
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_size * 3, self.vocab_size)
        self.pointer = nn.Linear(hidden_size * 3, 1)
        
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            embedded = self.embedding(input.unsqueeze(1))
        
        attention = self.attention(hidden[-1], encoder_outputs, mask)
        attention = attention.unsqueeze(1)
        
        weighted = torch.bmm(attention, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted), dim=1))
        p_gen = torch.sigmoid(self.pointer(torch.cat((output, weighted), dim=1)))
        
        return prediction, hidden, cell, p_gen, attention.squeeze(1)

class PointerGeneratorNetwork(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-chinese', hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        self.encoder = Encoder(self.bert, hidden_size, num_layers, dropout)
        self.decoder = Decoder(self.bert, hidden_size, num_layers, dropout)
        
    def forward(self, src: torch.Tensor, src_len: torch.Tensor, trg: torch.Tensor,
                src_mask: torch.Tensor) -> torch.Tensor:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        
        encoder_outputs, (hidden, cell) = self.encoder(src, src_len, src_mask)
        
        def _reshape_hidden(hidden):
            num_layers = hidden.size(0) // 2
            return hidden.view(2, num_layers, batch_size, -1).mean(0)
        
        hidden = _reshape_hidden(hidden)
        cell = _reshape_hidden(cell)
        
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(src.device)
        p_gens = torch.zeros(batch_size, trg_len, 1).to(src.device)
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(src.device)
        
        for t in range(trg_len):
            input = trg[:, t]
            prediction, hidden, cell, p_gen, attention = self.decoder(
                input, hidden, cell, encoder_outputs, src_mask)
            
            outputs[:, t] = prediction
            p_gens[:, t] = p_gen
            attentions[:, t] = attention
        
        return outputs, p_gens, attentions

class SummarizationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_src_len: int = 512, max_tgt_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        source = item['content']
        target = item['title']
        
        source_tokens = self.tokenizer.encode(source, max_length=self.max_src_len, truncation=True)
        target_tokens = self.tokenizer.encode(target, max_length=self.max_tgt_len, truncation=True)
        
        return {
            'source': torch.tensor(source_tokens),
            'target': torch.tensor(target_tokens),
            'source_len': len(source_tokens)
        }

def collate_fn(batch: List[Dict]) -> Dict:
    source_lens = [item['source_len'] for item in batch]
    max_source_len = max(source_lens)
    max_target_len = max(len(item['target']) for item in batch)
    
    sources = torch.zeros(len(batch), max_source_len).long()
    targets = torch.zeros(len(batch), max_target_len).long()
    source_mask = torch.zeros(len(batch), max_source_len).bool()
    
    for i, item in enumerate(batch):
        sources[i, :item['source_len']] = item['source']
        targets[i, :len(item['target'])] = item['target']
        source_mask[i, :item['source_len']] = 1
    
    return {
        'source': sources,
        'target': targets,
        'source_len': torch.tensor(source_lens),
        'source_mask': source_mask
    }

def train_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int) -> float:
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}')):
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        source_len = batch['source_len'].to(device)
        source_mask = batch['source_mask'].to(device)
        
        optimizer.zero_grad()
        
        output, p_gen, attention = model(source, source_len, target, source_mask)
        
        output = output[:, :-1].contiguous().view(-1, output.size(-1))
        target = target[:, 1:].contiguous().view(-1)
        
        loss = F.cross_entropy(output, target, ignore_index=0)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            source_len = batch['source_len'].to(device)
            source_mask = batch['source_mask'].to(device)
            
            output, p_gen, attention = model(source, source_len, target, source_mask)
            
            output = output[:, :-1].contiguous().view(-1, output.size(-1))
            target = target[:, 1:].contiguous().view(-1)
            
            loss = F.cross_entropy(output, target, ignore_index=0)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    with open('/d2/mxy/LLM-PEFT/NLP_Task/Summary_generation/data/lcsts_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:40000]
    print(f"Loaded {len(data)} samples")
    
    model = PointerGeneratorNetwork(
        bert_model_name='bert-base-chinese',
        hidden_size=256,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    dataset = SummarizationDataset(data, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    save_dir = '/d2/mxy/LLM-PEFT/NLP_Task/Summary_generation/models/PGN'
    os.makedirs(save_dir, exist_ok=True)
    
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_model(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate_model(model, val_loader, device)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f'Model saved with validation loss: {val_loss:.4f}')

if __name__ == "__main__":
    main()
