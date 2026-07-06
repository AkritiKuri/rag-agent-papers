"""
mlflow_experiments.py
Runs ingestion + a fixed evaluation query set under different configs
(chunk size, embedding model, retrieval-k) and logs params/metrics to
MLflow so you have real tradeoff data to show.

Usage:
    python -m src.mlflow_experiments
    mlflow ui   # then open http://localhost:5000
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids a Windows DLL conflict
                                              # between mlflow's deps and torch

import time

# torch is normally imported lazily (inside sentence-transformers, only
# when an embedding model is actually built). On Windows, if mlflow loads
# first, its numpy/protobuf build can leave a conflicting DLL in memory
# that crashes torch's init later. Forcing torch to import here, first,
# avoids that.
import torch  # noqa: F401

from src.ingestion import run_ingestion
from src.rag_core import RAGPipeline

import mlflow

EXPERIMENT_NAME = "rag-paper-qa"

# A handful of eval questions with a short "must mention" keyword to give
# a cheap, explainable retrieval-quality signal without needing labeled data.
EVAL_SET = [
    {"query": "What problem does this research address?", "must_mention": None},
    {"query": "What dataset or benchmark is used for evaluation?", "must_mention": None},
    {"query": "What are the main limitations discussed?", "must_mention": None},
]

CONFIGS = [
    {"chunk_size": 500, "chunk_overlap": 100, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "retrieval_k": 3},
    {"chunk_size": 1000, "chunk_overlap": 200, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "retrieval_k": 4},
    {"chunk_size": 1500, "chunk_overlap": 300, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "retrieval_k": 4},
    {"chunk_size": 1000, "chunk_overlap": 200, "embedding_model": "sentence-transformers/all-mpnet-base-v2", "retrieval_k": 4},
]


def run_single_config(config: dict) -> dict:
    with mlflow.start_run(run_name=f"cs{config['chunk_size']}_k{config['retrieval_k']}"):
        mlflow.log_params(config)

        t0 = time.time()
        stats = run_ingestion(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            embedding_model=config["embedding_model"],
            index_path=f"data/faiss_index_tmp_{config['chunk_size']}_{config['embedding_model'].split('/')[-1]}",
        )
        ingestion_time = time.time() - t0
        mlflow.log_metric("ingestion_time_sec", ingestion_time)
        mlflow.log_metric("num_chunks", stats["num_chunks"])

        pipeline = RAGPipeline(
            index_path=f"data/faiss_index_tmp_{config['chunk_size']}_{config['embedding_model'].split('/')[-1]}",
            embedding_model=config["embedding_model"],
            retrieval_k=config["retrieval_k"],
        )

        latencies, avg_scores = [], []
        for item in EVAL_SET:
            t1 = time.time()
            chunks = pipeline.retrieve(item["query"])
            latencies.append(time.time() - t1)
            if chunks:
                avg_scores.append(sum(c.score for c in chunks) / len(chunks))

        mlflow.log_metric("avg_retrieval_latency_sec", sum(latencies) / len(latencies))
        if avg_scores:
            mlflow.log_metric("avg_retrieval_distance", sum(avg_scores) / len(avg_scores))

        return {
            "config": config,
            "num_chunks": stats["num_chunks"],
            "ingestion_time_sec": round(ingestion_time, 2),
            "avg_retrieval_latency_sec": round(sum(latencies) / len(latencies), 4),
        }


def run_all():
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = []
    for config in CONFIGS:
        print(f"Running config: {config}")
        results.append(run_single_config(config))
    print("\nSummary:")
    for r in results:
        print(r)
    return results


if __name__ == "__main__":
    run_all()