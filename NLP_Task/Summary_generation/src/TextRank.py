import json
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re
from transformers import BertTokenizer, BertModel
import torch
from typing import List
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime

class TextRankSummarizer:
    def __init__(self, damping=0.85, min_diff=1e-5, steps=1000, bert_model_name='bert-base-chinese'):
        print("Initializing TextRankSummarizer...")
        self.damping = damping
        self.min_diff = min_diff
        self.steps = steps
        self.rouge = Rouge()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print("Loading BERT model and tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()
        print("Model initialization completed!")
        
    def _split_sentences(self, text: str) -> List[str]:
        sentence_ends = ['。', '！', '？', '…', '；', '\n']
        
        pattern = '|'.join([re.escape(sep) for sep in sentence_ends])
        sentences = re.split(f'({pattern})', text)
        
        merged_sentences = []
        for i in range(0, len(sentences)-1, 2):
            if sentences[i].strip():
                merged_sentences.append(sentences[i].strip() + (sentences[i+1] if i+1 < len(sentences) else ''))
        
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            merged_sentences.append(sentences[-1].strip())
            
        return merged_sentences

    def _get_bert_embeddings(self, sentences: List[str]) -> np.ndarray:
        embeddings = []
        
        with torch.no_grad():
            for sentence in tqdm(sentences, desc="Getting BERT embeddings", leave=False):
                inputs = self.tokenizer(sentence, 
                                      return_tensors='pt',
                                      padding=True, 
                                      truncation=True, 
                                      max_length=512).to(self.device)
                
                outputs = self.bert_model(**inputs)
                sentence_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                embeddings.append(sentence_embedding)
                
        return np.array(embeddings)

    def _build_similarity_matrix(self, vectors: np.ndarray) -> np.ndarray:
        similarity_matrix = cosine_similarity(vectors)
        
        norm = similarity_matrix.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1
        
        return similarity_matrix / norm

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return ''.join(sentences)

        vectors = self._get_bert_embeddings(sentences)
        
        similarity_matrix = self._build_similarity_matrix(vectors)
        
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, alpha=self.damping)
        
        ranked_sentences = [(scores[i], s, i) for i, s in enumerate(sentences)]
        ranked_sentences.sort(key=lambda x: x[0], reverse=True)
        
        selected_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[2])
        
        return ''.join([s[1] for s in selected_sentences])

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        rouge_scores = self.rouge.get_scores(hypothesis, reference)
        
        smooth = SmoothingFunction().method1
        reference_tokens = list(jieba.cut(reference))
        hypothesis_tokens = list(jieba.cut(hypothesis))
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth)
        
        return {
            'rouge': rouge_scores[0],
            'bleu': bleu_score
        }

def main():
    save_dir = '/d2/mxy/LLM-PEFT/NLP_Task/Summary_generation/models/TextRank'
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Loading dataset...")
    with open('/d2/mxy/LLM-PEFT/NLP_Task/Summary_generation/data/lcsts_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    summarizer = TextRankSummarizer()
    
    results = []
    
    for i, item in enumerate(tqdm(data[:2000], desc="Processing samples")):
        content = item['content']
        reference = item['title']
        
        summary = summarizer.summarize(content, num_sentences=1)
        scores = summarizer.evaluate(reference, summary)

        result = {
            'sample_id': i,
            'original_text': content,
            'reference_summary': reference,
            'generated_summary': summary,
            'rouge-1-p': scores['rouge']['rouge-1']['p'],
            'rouge-1-r': scores['rouge']['rouge-1']['r'],
            'rouge-1-f': scores['rouge']['rouge-1']['f'],
            'rouge-2-p': scores['rouge']['rouge-2']['p'],
            'rouge-2-r': scores['rouge']['rouge-2']['r'],
            'rouge-2-f': scores['rouge']['rouge-2']['f'],
            'rouge-l-p': scores['rouge']['rouge-l']['p'],
            'rouge-l-r': scores['rouge']['rouge-l']['r'],
            'rouge-l-f': scores['rouge']['rouge-l']['f'],
            'bleu': scores['bleu']
        }
        results.append(result)
        
        if i < 5:
            print(f"\nSample {i+1}:")
            print(f"Original text: {content}")
            print(f"Reference summary: {reference}")
            print(f"Generated summary: {summary}")
            print(f"ROUGE-1 F1: {scores['rouge']['rouge-1']['f']:.4f}")
            print(f"ROUGE-2 F1: {scores['rouge']['rouge-2']['f']:.4f}")
            print(f"ROUGE-L F1: {scores['rouge']['rouge-l']['f']:.4f}")
            print(f"BLEU: {scores['bleu']:.4f}")
    
    df = pd.DataFrame(results)
    results_file = os.path.join(save_dir, f'textrank_results_{timestamp}.csv')
    df.to_csv(results_file, index=False, encoding='utf-8')
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\nAverage Scores:")
    print(f"ROUGE-1: P={df['rouge-1-p'].mean():.4f}, R={df['rouge-1-r'].mean():.4f}, F1={df['rouge-1-f'].mean():.4f}")
    print(f"ROUGE-2: P={df['rouge-2-p'].mean():.4f}, R={df['rouge-2-r'].mean():.4f}, F1={df['rouge-2-f'].mean():.4f}")
    print(f"ROUGE-L: P={df['rouge-l-p'].mean():.4f}, R={df['rouge-l-r'].mean():.4f}, F1={df['rouge-l-f'].mean():.4f}")
    print(f"BLEU: {df['bleu'].mean():.4f}")
    
    summary_file = os.path.join(save_dir, f'textrank_summary_{timestamp}.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("TextRank Summarization Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of samples: {len(data)}\n\n")
        f.write("Average Scores:\n")
        f.write(f"ROUGE-1: P={df['rouge-1-p'].mean():.4f}, R={df['rouge-1-r'].mean():.4f}, F1={df['rouge-1-f'].mean():.4f}\n")
        f.write(f"ROUGE-2: P={df['rouge-2-p'].mean():.4f}, R={df['rouge-2-r'].mean():.4f}, F1={df['rouge-2-f'].mean():.4f}\n")
        f.write(f"ROUGE-L: P={df['rouge-l-p'].mean():.4f}, R={df['rouge-l-r'].mean():.4f}, F1={df['rouge-l-f'].mean():.4f}\n")
        f.write(f"BLEU: {df['bleu'].mean():.4f}\n")
    
    print(f"Summary results saved to: {summary_file}")

if __name__ == "__main__":
    main()
