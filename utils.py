import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from models import SequentialDataset, BERT4RecDataset

def create_data_loaders(X_train, y_train, X_val, y_val, num_items, model_type='gru', batch_sizes=(128, 256)):
    """Создание DataLoader'ов для разных моделей"""
    train_batch_size, val_batch_size = batch_sizes
    
    if model_type == 'bert':
        train_dataset = BERT4RecDataset(X_train, y_train, num_items=num_items)
        val_dataset = BERT4RecDataset(X_val, y_val, num_items=num_items)
    else:
        train_dataset = SequentialDataset(X_train, y_train, num_items=num_items)
        val_dataset = SequentialDataset(X_val, y_val, num_items=num_items)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def create_trainer(max_epochs=25, patience=5):
    """Создание тренера с callback'ами"""
    early_stopping = EarlyStopping(
        monitor='val_ndcg_10',
        patience=patience,
        mode='max',
        min_delta=0.001
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[early_stopping],
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    
    return trainer, early_stopping

def evaluate_model(model, test_loader, device):
    """Универсальная функция оценки модели"""
    # перемещаем модель на устройство
    model = model.to(device)
    model.eval()
    
    all_metrics = {k: {'hr': [], 'ndcg': []} for k in [5, 10, 20]}
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:  # BERT-style batch
                sequences, _, _, targets = batch
            else:  # GRU-style batch
                sequences, targets = batch
                
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)
                
            probs = torch.softmax(outputs, dim=1)
            
            for k in [5, 10, 20]:
                topk_probs, topk_indices = torch.topk(probs, k, dim=1)
                
                # Hit Rate
                hits = (topk_indices == targets.unsqueeze(1)).any(dim=1).float()
                all_metrics[k]['hr'].extend(hits.cpu().numpy())
                
                # NDCG
                ranks = torch.arange(1, k + 1, device=targets.device)
                discounts = 1.0 / torch.log2(ranks + 1)
                relevance = (topk_indices == targets.unsqueeze(1)).float()
                dcg = (relevance * discounts).sum(dim=1)
                ideal_relevance = torch.zeros_like(relevance)
                ideal_relevance[:, 0] = 1.0
                ideal_dcg = (ideal_relevance * discounts).sum(dim=1)
                ndcg = torch.where(ideal_dcg > 0, dcg / ideal_dcg, torch.zeros_like(dcg))
                all_metrics[k]['ndcg'].extend(ndcg.cpu().numpy())
    
    results = {}
    for k in [5, 10, 20]:
        results[k] = {
            'hit_rate': np.mean(all_metrics[k]['hr']),
            'ndcg': np.mean(all_metrics[k]['ndcg'])
        }
    
    return results

def save_model(model, model_config, vocabulary, metrics, filepath):
    """Сохранение модели и метрик"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'vocabulary': vocabulary,
        'metrics': metrics
    }, filepath)
    print(f"Модель сохранена в: {filepath}")

def print_training_summary(trainer, early_stopping, metrics, model_name):
    """Вывод итогов обучения"""
    print(f"ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ {model_name}")
    print(f"Обучено эпох: {trainer.current_epoch}")
    print(f"Ранняя остановка: {'Да' if early_stopping.stopped_epoch > 0 else 'Нет'}")
    
    for k in [5, 10, 20]:
        print(f"Top-{k}: HR={metrics[k]['hit_rate']:.4f}, NDCG={metrics[k]['ndcg']:.4f}")

def compare_models(gru_metrics, bert_metrics, model_names=("GRU4Rec", "BERT4Rec")):
    """Сравнение метрик моделей"""
    print(f"\nСравнение {model_names[0]} vs {model_names[1]}:")

    
    for k in [5, 10, 20]:
        gru_hr = gru_metrics[k]['hit_rate']
        bert_hr = bert_metrics[k]['hit_rate']
        hr_diff = bert_hr - gru_hr
        
        gru_ndcg = gru_metrics[k]['ndcg']
        bert_ndcg = bert_metrics[k]['ndcg']
        ndcg_diff = bert_ndcg - gru_ndcg
        
        print(f"Top-{k}:")
        print(f"  HR:    {model_names[0]}={gru_hr:.4f} vs {model_names[1]}={bert_hr:.4f} (diff: {hr_diff:+.4f})")
        print(f"  NDCG:  {model_names[0]}={gru_ndcg:.4f} vs {model_names[1]}={bert_ndcg:.4f} (diff: {ndcg_diff:+.4f})")