import re
from collections import Counter
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

# =====================================
# Modelo de n-grams para previsão de palavras
# =====================================
class NgramModel:
    """Modelo de linguagem baseado em n-grams para previsão de palavras"""
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = None
        self.nminus1_counts = None

    def tokenize(self, text):
        """Tokeniza o texto removendo pontuações e normalizando para minúsculas"""
        return re.findall(r'\w+', text.lower())

    def build_ngram_counts(self, tokens, n):
        """Constrói contagens de n-grams a partir de tokens"""
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    def build_models(self, text):
        """Constrói os modelos de n-gram e (n-1)-gram"""
        tokens = self.tokenize(text)
        self.ngram_counts = self.build_ngram_counts(tokens, self.n)
        self.nminus1_counts = self.build_ngram_counts(tokens, self.n-1) if self.n > 1 else None

    def predict_next_word(self, prefix):
        """Prevê a próxima palavra com base no prefixo"""
        if self.n == 1:
            total = sum(self.ngram_counts.values())
            return {word[0]: count/total for word, count in self.ngram_counts.items()}
        
        if self.nminus1_counts is None:
            return {}

        total = self.nminus1_counts.get(prefix, 0)
        if total == 0:
            return {}

        candidates = {}
        for ngram, count in self.ngram_counts.items():
            if ngram[:-1] == prefix:
                candidates[ngram[-1]] = count / total
        return candidates

# =====================================
# Implementação do Transformer
# =====================================
class PositionalEncoding(nn.Module):
    """Codificação posicional para transformers"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """Modelo Transformer para previsão de palavras"""
    def __init__(self, vocab_size, d_model=128, n_heads=8, num_layers=2, d_ff=256, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            self.TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    class TransformerBlock(nn.Module):
        """Bloco Transformer com atenção multi-head e feed-forward"""
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, n_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.ff(x)
            return self.norm2(x + self.dropout(ff_output))

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        # Para cada camada do Transformer
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

# =====================================
# Interface de Chat
# =====================================
class ChatInterface:
    def __init__(self, model, vocab, seq_length=20, max_response_length=50):
        self.model = model
        self.vocab = vocab  # Contém idx2word e word2idx
        self.seq_length = seq_length
        self.max_response_length = max_response_length
        self.history = []
        
    def generate_response(self, prompt, temperature=0.7):
        self.model.eval()
        # Tokenização simples do prompt
        tokens = [self.vocab.word2idx.get(word, 1) for word in prompt.lower().split()]
        
        with torch.no_grad():
            for _ in range(self.max_response_length):
                inputs = torch.tensor(tokens[-self.seq_length:]).unsqueeze(0)
                outputs = self.model(inputs)
                probs = F.softmax(outputs[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Critério de parada: token de <pad> ou pontuações finais
                if next_token == 0 or (self.vocab.idx2word.get(next_token, '') in ['.', '?', '!']):
                    break
                tokens.append(next_token)
        
        # Gera resposta utilizando os tokens que não fazem parte do prompt original
        response_words = []
        for idx in tokens[len(prompt.split()):]:
            word = self.vocab.idx2word.get(idx, '')
            if word not in ['<pad>', '<unk>']:
                response_words.append(word)
        
        return self.postprocess_response(' '.join(response_words))
    
    def postprocess_response(self, response):
        # Corrige a formatação e remove duplicados
        response = ' '.join([word for i, word in enumerate(response.split()) 
                             if word not in response.split()[:i]])
        return response.capitalize()
    
    def start_chat(self):
        print("\n=== Modo Conversa ===")
        print("Digite sua mensagem (ou 'sair' para encerrar)")
        while True:
            try:
                user_input = input("\nVocê: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    break
                
                response = self.generate_response(user_input)
                print(f"\nModelo: {response}")
                self.history.append((user_input, response))
                
            except Exception as e:
                print(f"\nErro na geração: {str(e)}")

# =====================================
# Classe para Treinamento do Modelo de Linguagem
# =====================================
class LanguageModelTrainer:
    """Classe para treinamento e fine-tuning de modelos de linguagem"""
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_alignment_loss(self, pretrained_state, lambda_reg=0.01):
        """Calcula a perda de alinhamento com pesos pré-treinados"""
        loss = 0.0
        for name, param in self.model.named_parameters():
            loss += torch.sum((param - pretrained_state[name]) ** 2)
        return lambda_reg * loss  # Fora do loop

    def train(self, dataset, num_epochs, pretrained_state=None, lambda_reg=0.0):
        """Treina o modelo com opção de alinhamento com pesos pré-treinados"""
        for epoch in range(num_epochs):
            total_loss = 0.0
            for inputs, targets in dataset:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                if pretrained_state and lambda_reg > 0:
                    loss += self.compute_alignment_loss(pretrained_state, lambda_reg)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Época {epoch+1}: Loss {total_loss/len(dataset):.4f}")

# =====================================
# Vocabulário
# =====================================
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def build(self, text):
        tokens = re.findall(r'\w+', text.lower())
        counter = Counter(tokens)
        sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.idx = 2
        
        for word, _ in sorted_words:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def encode(self, text):
        tokens = re.findall(r'\w+', text.lower())
        return [self.word2idx.get(token, 1) for token in tokens]

# =====================================
# Função para gerar previsões (não utilizada na interface de chat)
# =====================================
def generate_text(model, vocab, initial_text, max_length=20, temperature=0.7):
    model.eval()
    tokens = vocab.encode(initial_text)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Use a variável seq_length definida no fluxo principal
            inputs = torch.tensor(tokens[-seq_length:]).unsqueeze(0)
            outputs = model(inputs)
            
            # Aplica temperatura
            logits = outputs[0, -1] / temperature
            probabilities = F.softmax(logits, dim=-1)
            
            # Amostragem a partir das probabilidades
            next_token = torch.multinomial(probabilities, 1).item()
            tokens.append(next_token)
            
            if next_token == 0:  # <pad> detectado
                break
    
    return ' '.join([vocab.idx2word.get(idx, '<unk>') for idx in tokens])

# =====================================
# Função para carregar dados do JSON considerando o dataset de QA
# =====================================
def load_texts_from_json(file_path):  
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = []
        # Verifica se os dados estão dentro da chave "qa_dataset"
        if isinstance(data, dict) and "qa_dataset" in data:
            for item in data["qa_dataset"]:
                # Concatena pergunta e resposta para formar o corpus de treinamento
                question = item.get("question", "")
                answer = item.get("answer", "")
                texts.append(f"{question} {answer}")
        # Caso o JSON seja uma lista de dicionários com a chave "texto"
        elif isinstance(data, list):
            texts = [item.get('texto', '') for item in data if isinstance(item, dict)]
        else:
            print("Formato de JSON não reconhecido.")
            return ""
        
        if not texts:
            print("Nenhum texto encontrado no JSON!")
            return ""
        return ' '.join(texts)
    except json.JSONDecodeError as e:
        print(f"Erro ao carregar JSON: {str(e)}")
        return ""
    except Exception as e:
        print(f"Erro: {str(e)}")
        return ""

# =====================================
# Função de validação e tokenização do texto de treinamento
# =====================================
def validate_and_tokenize(text, min_text_length):
    tokens = re.findall(r'\w+', text.lower())
    if len(tokens) < min_text_length:
        print(f"Texto precisa de pelo menos {min_text_length} palavras (encontradas: {len(tokens)})")
        return None, None
    return text, tokens

# =====================================
# Código Principal
# =====================================
if __name__ == "__main__":
    # Configurações
    seq_length = 64
    num_epochs = 1000
    min_text_length = seq_length + 1
    json_path = "dataset.json"  # Certifique-se de que o arquivo JSON está no formato esperado

    # 1. Carregar dados do JSON
    text = load_texts_from_json(json_path)
    if not text:
        exit("Nenhum texto carregado do JSON!")
    
    validated_text, tokens = validate_and_tokenize(text, min_text_length)
    if not validated_text:
        exit("Texto insuficiente para treinamento!")
    
    # 2. Construir vocabulário usando os tokens do corpus
    class EnhancedVocabulary:
        def __init__(self):
            self.word2idx = {'<pad>': 0, '<unk>': 1}
            self.idx2word = {0: '<pad>', 1: '<unk>'}
            
        def build(self, tokens):
            counter = Counter(tokens)
            for word, _ in counter.most_common():
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
            print(f"Vocabulário criado com {len(self.word2idx)} palavras")

    vocab = EnhancedVocabulary()
    vocab.build(tokens)
    vocab_size = len(vocab.word2idx)

    # 3. Preparar dados de treino (sequências de tokens)
    def prepare_data():
        encoded = [vocab.word2idx.get(word, 1) for word in tokens]
        sequences = []
        for i in range(len(encoded) - seq_length):
            # Cada sequência tem uma entrada e um alvo deslocado em 1 token
            sequences.append((
                torch.tensor(encoded[i:i+seq_length]),
                torch.tensor(encoded[i+1:i+1+seq_length])
            ))
        return sequences
    
    try:
        train_data = prepare_data()
        print(f"Geradas {len(train_data)} sequências de treino")
    except Exception as e:
        exit(f"Erro na preparação de dados: {str(e)}")


    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        num_layers=2,
        max_seq_length=seq_length
    )
    
    try:
        trainer = LanguageModelTrainer(transformer)
        print("\n=== Iniciando Treinamento ===")
        trainer.train(train_data, num_epochs=num_epochs)
    except Exception as e:
        exit(f"Erro no treinamento: {str(e)}")

    
    chat = ChatInterface(
        model=transformer,
        vocab=vocab,
        seq_length=seq_length
    )
    chat.start_chat()