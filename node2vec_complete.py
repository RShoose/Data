# node2vec_complete.py (обновленная версия)
import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

class Node2VecComplete(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(Node2VecComplete, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Раздельные эмбеддинги для центральных и контекстных узлов
        self.center_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Инициализация по схеме Word2Vec
        init_range = 1.0 / embedding_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_nodes, context_nodes, negative_nodes):
        center_embeds = self.center_embeddings(center_nodes)
        context_embeds = self.context_embeddings(context_nodes)
        negative_embeds = self.context_embeddings(negative_nodes)
        
        return center_embeds, context_embeds, negative_embeds

    def compute_loss(self, center_embeds, context_embeds, negative_embeds):
        # Положительные примеры
        positive_scores = torch.sum(center_embeds * context_embeds, dim=1)
        positive_loss = F.logsigmoid(positive_scores)
        
        # Отрицательные примеры
        negative_scores = torch.bmm(negative_embeds, center_embeds.unsqueeze(2))
        negative_scores = negative_scores.squeeze(2)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        
        # Negative Sampling Loss
        loss = - (positive_loss + negative_loss).mean()
        return loss

class CompleteNode2Vec:
    def __init__(self, G, dimensions=128, walk_length=80, num_walks=10, 
                 window_size=10, num_negatives=5, p=1.0, q=1.0):
        self.G = G
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.p = p
        self.q = q
        
        # Создаем mapping узлов
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)
        
        print(f"Инициализирован CompleteNode2Vec для графа с {self.num_nodes} узлами")
        
        # Предвычисляем вероятности переходов с учетом весов
        self.alias_nodes = {}
        self.precompute_transition_probs()

    def precompute_transition_probs(self):
        """Предвычисление вероятностей переходов с учетом весов ребер"""
        print("Предвычисление вероятностей переходов...")
        
        successful_nodes = 0
        for node in self.G.nodes():
            try:
                neighbors = list(self.G.neighbors(node))
                if not neighbors:
                    self.alias_nodes[node] = None
                    continue
                
                # Получаем веса ребер с обработкой ошибок
                weights = []
                for neighbor in neighbors:
                    try:
                        weight = self.G[node][neighbor].get('weight', 1.0)
                        weights.append(weight)
                    except KeyError:
                        # Если ребра нет, используем вес по умолчанию
                        weights.append(1.0)
                
                # Нормализация весов для вероятностей
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_probs = [w / total_weight for w in weights]
                    self.alias_nodes[node] = self.create_alias_table(normalized_probs)
                    successful_nodes += 1
                else:
                    self.alias_nodes[node] = None
                    
            except Exception as e:
                print(f"Ошибка при обработке узла {node}: {e}")
                self.alias_nodes[node] = None
        
        print(f"Успешно обработано {successful_nodes}/{len(self.G.nodes())} узлов")

    def create_alias_table(self, probs):
        """Создание alias таблицы для O(1) сэмплирования"""
        try:
            K = len(probs)
            q = np.array(probs) * K
            smaller, larger = [], []
            
            for i, q_i in enumerate(q):
                if q_i < 1.0:
                    smaller.append(i)
                else:
                    larger.append(i)
                    
            alias_table = np.zeros(K, dtype=np.int32)
            prob_table = np.zeros(K)
            
            while smaller and larger:
                small = smaller.pop()
                large = larger.pop()
                
                alias_table[small] = large
                prob_table[small] = q[small]
                
                q[large] = q[large] - (1.0 - q[small])
                if q[large] < 1.0:
                    smaller.append(large)
                else:
                    larger.append(large)
                    
            for i in larger + smaller:
                prob_table[i] = 1.0
                
            return prob_table, alias_table
        except Exception as e:
            print(f"Ошибка при создании alias таблицы: {e}")
            return None

    def alias_draw(self, alias_table):
        """Сэмплирование из alias таблицы"""
        if alias_table is None:
            return 0
        try:
            J = len(alias_table[0])
            k = int(np.floor(np.random.rand() * J))
            return k if np.random.rand() < alias_table[0][k] else alias_table[1][k]
        except:
            return 0

    def node2vec_walk(self, start_node):
        """Генерация одного блуждания с использованием Node2Vec стратегии"""
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            try:
                neighbors = list(self.G.neighbors(cur))
                if not neighbors:
                    break
                    
                if len(walk) == 1:
                    # Первый шаг - сэмплирование по весам
                    if self.alias_nodes.get(cur) is not None:
                        alias_table = self.alias_nodes[cur]
                        if alias_table is not None:
                            idx = self.alias_draw(alias_table)
                            if idx < len(neighbors):
                                next_node = neighbors[idx]
                            else:
                                next_node = random.choice(neighbors)
                        else:
                            next_node = random.choice(neighbors)
                    else:
                        next_node = random.choice(neighbors)
                else:
                    # Node2Vec стратегия с параметрами p и q
                    prev = walk[-2]
                    next_node = self.get_next_node(prev, cur)
                    
                if next_node and next_node in self.G.nodes():
                    walk.append(next_node)
                else:
                    break
                    
            except Exception as e:
                break
                
        return walk

    def get_next_node(self, prev, cur):
        """Выбор следующего узла с учетом параметров p и q и весов ребер"""
        try:
            neighbors = list(self.G.neighbors(cur))
            if not neighbors:
                return None
                
            unnormalized_probs = []
            for neighbor in neighbors:
                try:
                    weight = self.G[cur][neighbor].get('weight', 1.0)
                    
                    if neighbor == prev:
                        unnormalized_probs.append(weight / self.p)
                    elif self.G.has_edge(prev, neighbor):
                        unnormalized_probs.append(weight)
                    else:
                        unnormalized_probs.append(weight / self.q)
                except KeyError:
                    # Если ребра нет, используем вес по умолчанию
                    unnormalized_probs.append(1.0)
            
            total = sum(unnormalized_probs)
            if total == 0:
                return random.choice(neighbors)
                
            normalized_probs = [p / total for p in unnormalized_probs]
            
            # Сэмплирование с использованием alias method
            if self.alias_nodes.get(cur) is not None:
                temp_alias_table = self.create_alias_table(normalized_probs)
                if temp_alias_table is not None:
                    idx = self.alias_draw(temp_alias_table)
                    if idx < len(neighbors):
                        return neighbors[idx]
            
            return np.random.choice(neighbors, p=normalized_probs)
            
        except Exception as e:
            return random.choice(list(self.G.neighbors(cur)))

    def generate_training_data(self):
        """Генерация тренировочных данных из случайных блужданий"""
        print("Генерация тренировочных данных...")
        all_pairs = []
        
        successful_walks = 0
        total_walks = self.num_walks * len(self.nodes)
        
        for walk_num in range(self.num_walks):
            nodes = list(self.G.nodes())
            random.shuffle(nodes)
            
            for start_node in nodes:
                try:
                    walk = self.node2vec_walk(start_node)
                    if len(walk) > 1:
                        walk_indices = [self.node_to_idx[node] for node in walk]
                        
                        # Создание пар (center, context) для Skip-gram
                        for i, center_idx in enumerate(walk_indices):
                            start = max(0, i - self.window_size)
                            end = min(len(walk_indices), i + self.window_size + 1)
                            
                            for j in range(start, end):
                                if j != i:
                                    context_idx = walk_indices[j]
                                    # Учитываем вес связи при создании multiple примеров
                                    try:
                                        center_node = self.idx_to_node[center_idx]
                                        context_node = self.idx_to_node[context_idx]
                                        weight = self.G[center_node][context_node].get('weight', 1.0)
                                        
                                        # Создаем больше примеров для связей с большим весом
                                        num_examples = max(1, int(weight))
                                        for _ in range(num_examples):
                                            all_pairs.append((center_idx, context_idx))
                                    except:
                                        # Если есть ошибка, добавляем один пример
                                        all_pairs.append((center_idx, context_idx))
                        
                        successful_walks += 1
                except Exception as e:
                    continue
        
        print(f"Успешно сгенерировано {successful_walks}/{total_walks} блужданий")
        print(f"Создано {len(all_pairs)} тренировочных пар")
        return all_pairs

    def train(self, learning_rate=0.025, epochs=5, batch_size=128, device='cpu'):
        """Обучение модели с мониторингом прогресса"""
        print("Начало обучения CompleteNode2Vec...")
        
        # Генерация тренировочных данных
        training_pairs = self.generate_training_data()
        
        if not training_pairs:
            print("Ошибка: не удалось сгенерировать тренировочные данные")
            return {}
        
        # Инициализация модели
        model = Node2VecComplete(self.num_nodes, self.dimensions).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Подготовка данных
        try:
            center_nodes = torch.tensor([pair[0] for pair in training_pairs], dtype=torch.long).to(device)
            context_nodes = torch.tensor([pair[1] for pair in training_pairs], dtype=torch.long).to(device)
        except Exception as e:
            print(f"Ошибка при подготовке данных: {e}")
            return {}

        # Обучение
        model.train()
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Случайное перемешивание данных
            indices = torch.randperm(len(center_nodes))
            
            for i in range(0, len(indices), batch_size):
                try:
                    batch_indices = indices[i:i + batch_size]
                    
                    batch_center = center_nodes[batch_indices]
                    batch_context = context_nodes[batch_indices]
                    
                    # Генерация отрицательных примеров
                    batch_negative = torch.randint(0, self.num_nodes, 
                                                 (len(batch_center), self.num_negatives)).to(device)
                    
                    # Forward pass
                    center_emb, context_emb, negative_emb = model(batch_center, batch_context, batch_negative)
                    
                    # Вычисление loss
                    loss = model.compute_loss(center_emb, context_emb, negative_emb)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            losses.append(avg_loss)
            print(f"Эпоха {epoch+1}/{epochs}, Средний Loss: {avg_loss:.4f}")
        
        # Визуализация процесса обучения
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, epochs + 1), losses, marker='o', linewidth=2)
        plt.xlabel('Эпоха')
        plt.ylabel('Loss')
        plt.title('Процесс обучения CompleteNode2Vec')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Извлечение финальных эмбеддингов
        embeddings = {}
        with torch.no_grad():
            all_embeddings = model.center_embeddings.weight.cpu().numpy()
            for idx, node in self.idx_to_node.items():
                embeddings[node] = all_embeddings[idx]
        
        print("Обучение CompleteNode2Vec завершено!")
        return embeddings