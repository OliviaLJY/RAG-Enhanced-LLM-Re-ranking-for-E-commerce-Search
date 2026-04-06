"""
BM25 Baseline for MS MARCO Passage Ranking
===========================================
Day 1-2 目标：构建 BM25 检索 baseline

输入：query
输出：top-100 文档（BM25排序）

这是论文的 Baseline #1
"""

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from typing import List, Dict, Set, Tuple
import json
import os


class BM25Retriever:
    """BM25 检索器"""
    
    def __init__(self, corpus: List[str] = None):
        self.corpus = corpus or []
        self.tokenized_corpus = []
        self.bm25 = None
        
    def build_index(self, corpus: List[str]):
        """构建 BM25 索引"""
        print("📦 Building BM25 index...")
        self.corpus = corpus
        # 简单 tokenization（split by space）
        self.tokenized_corpus = [doc.lower().split() for doc in tqdm(corpus, desc="Tokenizing")]
        print(f"✅ Corpus size: {len(self.corpus):,} passages")
        
        print("⚙️ Initializing BM25...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("✅ BM25 index ready!")
        
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[int, str, float]]:
        """
        检索 top-k 文档
        
        Returns:
            List of (doc_id, passage_text, score)
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), self.corpus[idx], float(scores[idx])))
        
        return results


def load_msmarco_data(split: str = "train", limit: int = None):
    """
    加载 MS MARCO 数据
    
    Args:
        split: "train" 或 "validation"
        limit: 限制样本数（用于测试，避免太慢）
    """
    print(f"📥 Loading MS MARCO {split} data...")
    # 使用 microsoft/ms_marco-v1.1 官方数据集
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=split)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"⚠️ Limited to {limit} samples for testing")
    
    print(f"✅ Loaded {len(dataset):,} queries")
    return dataset


def build_corpus_from_dataset(dataset) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    从数据集构建 corpus
    
    Returns:
        corpus: 所有 passage 的列表
        query_to_relevant: query_idx -> [relevant_passage_indices]
    """
    print("🔧 Building corpus from dataset...")
    corpus = []
    passage_to_idx = {}  # passage_text -> corpus_idx（用于去重）
    query_to_relevant = {}  # query_idx -> [relevant_passage_corpus_idx]
    
    for query_idx, item in enumerate(tqdm(dataset, desc="Processing")):
        passages = item['passages']['passage_text']
        is_selected = item['passages']['is_selected']
        
        relevant_indices = []
        
        for passage, selected in zip(passages, is_selected):
            # 去重：检查 passage 是否已存在
            if passage not in passage_to_idx:
                corpus_idx = len(corpus)
                corpus.append(passage)
                passage_to_idx[passage] = corpus_idx
            else:
                corpus_idx = passage_to_idx[passage]
            
            # 记录相关文档
            if selected == 1:
                relevant_indices.append(corpus_idx)
        
        query_to_relevant[query_idx] = relevant_indices
    
    print(f"✅ Corpus built: {len(corpus):,} unique passages")
    return corpus, query_to_relevant


# ============== 评估指标 ==============

def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    """计算 Recall@k"""
    if len(relevant_ids) == 0:
        return 0.0
    
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(
    retriever: BM25Retriever,
    dataset,
    query_to_relevant: Dict[int, List[int]],
    top_k: int = 100,
    num_eval: int = None
) -> Dict[str, float]:
    """
    评估检索器性能
    
    Returns:
        metrics: {'Recall@10': ..., 'Recall@100': ..., 'MRR@10': ...}
    """
    print(f"\n📊 Evaluating retriever on {num_eval or len(dataset)} queries...")
    
    recall_10_scores = []
    recall_100_scores = []
    mrr_10_scores = []
    
    eval_indices = range(min(num_eval, len(dataset))) if num_eval else range(len(dataset))
    
    for query_idx in tqdm(eval_indices, desc="Evaluating"):
        query = dataset[query_idx]['query']
        relevant_ids = set(query_to_relevant.get(query_idx, []))
        
        # 跳过没有相关文档的 query
        if len(relevant_ids) == 0:
            continue
        
        # 检索
        results = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r[0] for r in results]
        
        # 计算指标
        recall_10_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=10))
        recall_100_scores.append(recall_at_k(retrieved_ids, relevant_ids, k=100))
        mrr_10_scores.append(mrr(retrieved_ids[:10], relevant_ids))
    
    metrics = {
        'Recall@10': np.mean(recall_10_scores) if recall_10_scores else 0.0,
        'Recall@100': np.mean(recall_100_scores) if recall_100_scores else 0.0,
        'MRR@10': np.mean(mrr_10_scores) if mrr_10_scores else 0.0,
        'Num_Queries': len(recall_10_scores)
    }
    
    return metrics


def demo_single_query(retriever: BM25Retriever, query: str, top_k: int = 5):
    """演示单个 query 的检索结果"""
    print(f"\n🔍 Query: {query}")
    print("-" * 60)
    
    results = retriever.retrieve(query, top_k=top_k)
    
    for rank, (doc_id, passage, score) in enumerate(results, start=1):
        # 截断过长的 passage
        display_text = passage[:200] + "..." if len(passage) > 200 else passage
        print(f"[{rank}] (score: {score:.4f})")
        print(f"    {display_text}")
        print()


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 BM25 Baseline for MS MARCO Passage Ranking")
    print("=" * 60)
    
    # ========== Step 1: 加载数据 ==========
    # 先用小数据集测试（5000 samples）
    # 完整实验时可以去掉 limit
    SAMPLE_LIMIT = 5000  # 设为 None 用全部数据
    
    dataset = load_msmarco_data(split="train", limit=SAMPLE_LIMIT)
    
    # ========== Step 2: 构建 corpus ==========
    corpus, query_to_relevant = build_corpus_from_dataset(dataset)
    
    # ========== Step 3: 构建 BM25 索引 ==========
    retriever = BM25Retriever()
    retriever.build_index(corpus)
    
    # ========== Step 4: 单 query 演示 ==========
    print("\n" + "=" * 60)
    print("📝 Demo: Single Query Retrieval")
    print("=" * 60)
    
    # 用数据集中的第一个 query 演示
    sample_query = dataset[0]['query']
    demo_single_query(retriever, sample_query, top_k=5)
    
    # 也试一个自定义 query
    demo_single_query(retriever, "best headphones for gym", top_k=5)
    
    # ========== Step 5: 评估 ==========
    print("\n" + "=" * 60)
    print("📊 Evaluation")
    print("=" * 60)
    
    # 评估（限制数量加快速度）
    NUM_EVAL = 500  # 评估 500 个 query
    
    metrics = evaluate_retriever(
        retriever,
        dataset,
        query_to_relevant,
        top_k=100,
        num_eval=NUM_EVAL
    )
    
    # ========== 输出结果 ==========
    print("\n" + "=" * 60)
    print("📈 BM25 Baseline Results")
    print("=" * 60)
    print(f"  Evaluated Queries: {metrics['Num_Queries']}")
    print(f"  Recall@10:  {metrics['Recall@10']:.4f}")
    print(f"  Recall@100: {metrics['Recall@100']:.4f}")
    print(f"  MRR@10:     {metrics['MRR@10']:.4f}")
    print("=" * 60)
    
    # 保存结果
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "bm25_baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'model': 'BM25 (rank_bm25)',
            'dataset': 'MS MARCO v1.1',
            'corpus_size': len(corpus),
            'metrics': metrics,
            'config': {
                'sample_limit': SAMPLE_LIMIT,
                'num_eval': NUM_EVAL
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Day 1-2 BM25 Baseline 完成！")
    print("👉 下一步：接入 LLM Re-ranking")


if __name__ == "__main__":
    main()
