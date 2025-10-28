import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SequentialDataset(Dataset):
    """Базовый класс для последовательных данных"""
    def __init__(self, sequences, targets, num_items=None):
        self.sequences = sequences
        self.targets = targets
        self.num_items = num_items
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.LongTensor(self.sequences[idx])
        target = torch.LongTensor([self.targets[idx]]).squeeze()
        return sequence, target

class BERT4RecDataset(SequentialDataset):
    """Датасет для BERT4Rec с маскированием"""
    def __init__(self, sequences, targets, mask_prob=0.15, num_items=None):
        super().__init__(sequences, targets, num_items)
        self.mask_prob = mask_prob
        self.mask_token = num_items
        
    def __getitem__(self, idx):
        sequence, target = super().__getitem__(idx)
        
        # маскирование
        masked_sequence = sequence.clone()
        masked_positions = torch.zeros(3, dtype=torch.long)
        masked_targets = torch.zeros(3, dtype=torch.long)
        
        return masked_sequence, masked_positions, masked_targets, target

class BaseRecModel(pl.LightningModule):
    """Базовый класс для рекомендательных моделей"""
    def __init__(self, num_items, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def _calculate_metrics(self, outputs, targets):
        """Расчет метрик HR и NDCG"""
        probs = torch.softmax(outputs, dim=1)
        metrics = {}
        
        for k in [5, 10, 20]:
            topk_probs, topk_indices = torch.topk(probs, k, dim=1)
            
            # Hit Rate
            hits = (topk_indices == targets.unsqueeze(1)).any(dim=1).float()
            hit_rate = hits.mean()
            
            # NDCG
            ranks = torch.arange(1, k + 1, device=targets.device)
            discounts = 1.0 / torch.log2(ranks + 1)
            relevance = (topk_indices == targets.unsqueeze(1)).float()
            dcg = (relevance * discounts).sum(dim=1)
            ideal_relevance = torch.zeros_like(relevance)
            ideal_relevance[:, 0] = 1.0
            ideal_dcg = (ideal_relevance * discounts).sum(dim=1)
            ndcg = (dcg / ideal_dcg).mean()
            
            metrics[f'hr_{k}'] = hit_rate
            metrics[f'ndcg_{k}'] = ndcg
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer

class ImprovedBERT4Rec(nn.Module):
    def __init__(self, num_items, max_len=20, hidden_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.num_items = num_items + 1  # +1 для MASK токена
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        
        # Item embeddings
        self.item_embedding = nn.Embedding(self.num_items, hidden_dim, padding_idx=0)
        
        # Position embeddings
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # нормализация
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, num_items)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences, masked_positions=None):
        batch_size, seq_len = sequences.shape
        
        # эмбеддинги
        item_emb = self.item_embedding(sequences)
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        embeddings = item_emb + pos_emb
        embeddings = self.dropout(embeddings)
        
        # Transformer
        transformer_out = self.transformer(embeddings)
        transformer_out = self.layer_norm(transformer_out)
        
        # предсказываем следующий товар
        output = self.output(transformer_out[:, -1, :])
        
        return output

class FixedBERT4RecModel(BaseRecModel):
    def __init__(self, num_items, learning_rate=1e-3, mask_prob=0.15):
        super().__init__(num_items, learning_rate)
        self.mask_prob = mask_prob
        
        self.model = ImprovedBERT4Rec(
            num_items=num_items,
            hidden_dim=64,
            nhead=4,
            num_layers=2
        )
        
    def forward(self, x, masked_positions=None):
        return self.model(x, masked_positions)
    
    def training_step(self, batch, batch_idx):
        sequences, _, _, targets = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, _, _, targets = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, targets)
        
        metrics = self._calculate_metrics(outputs, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
        
        return loss

class GRU4Rec(nn.Module):
    def __init__(self, num_items, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.item_embedding = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.output = nn.Linear(hidden_dim, num_items)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, sequences):
        embedded = self.item_embedding(sequences)
        embedded = self.dropout(embedded)
        
        gru_out, _ = self.gru(embedded)
        last_output = gru_out[:, -1, :]
        
        output = self.output(last_output)
        return output

class GRU4RecModel(BaseRecModel):
    def __init__(self, num_items, hidden_dim=64, learning_rate=1e-3):
        super().__init__(num_items, learning_rate)
        self.model = GRU4Rec(num_items, hidden_dim)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, targets)
        
        metrics = self._calculate_metrics(outputs, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
        
        return loss
