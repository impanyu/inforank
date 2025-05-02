import json
from trust_framework import TrustFramework
from typing import List, Dict
import os
from datasets import load_dataset
import random

def load_dataset(dataset_path: str) -> List[str]:
    """Load texts from dataset file"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data['texts']

def load_queries(queries_path: str) -> List[str]:
    """Load test queries"""
    with open(queries_path, 'r') as f:
        data = json.load(f)
    return data['queries']

def load_real_dataset(num_samples: int = 100) -> tuple[List[str], List[str]]:
    """Load real dataset from HuggingFace datasets"""
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    
    # Sample random articles
    samples = random.sample(range(len(dataset)), num_samples)
    
    texts = []
    queries = []
    for idx in samples:
        article = dataset[idx]
        # Get a paragraph from the article
        paragraphs = [p for p in article['text'].split('\n') if len(p) > 100]
        if paragraphs:
            texts.append(random.choice(paragraphs))
            queries.append(article['title'])
    
    return texts, queries

def main():
    # Initialize framework
    framework = TrustFramework(
        minimal_trustworthiness=0.2,
        consistency_threshold=0.5,
        credulity_factor=0.5,
        db_path="./test_db"
    )
    
    # Check if database already exists
    if os.path.exists("./test_db/faiss_index.bin"):
        print("Loading existing database...")
        if framework.load_db():
            print("Database loaded successfully")
        else:
            print("Failed to load database, starting fresh")
    
    # Load dataset and queries
    try:
        texts, queries = load_real_dataset(num_samples=10)  # Start with 10 samples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to sample data...")
        texts = [
            "The Earth orbits around the Sun.",
            "Water is composed of hydrogen and oxygen.",
            "Python is a programming language.",
        ]
        queries = [
            "What orbits the Sun?",
            "Tell me about water composition",
        ]
    
    # Populate database
    print(f"\nPopulating database with {len(texts)} texts...")
    for i, text in enumerate(texts, 1):
        result = framework.input(text)
        print(f"Processed text {i}/{len(texts)}: {result}")
    
    # Save database
    print("\nSaving database...")
    if framework.write_db():
        print("Database saved successfully")
    else:
        print("Failed to save database")
    
    # Test queries
    print(f"\nTesting {len(queries)} queries...")
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = framework.retrieve(query)
        print(f"Retrieved text: {result['text']}")
        print(f"Trust score: {result['trust_score']:.2f}")
        print(f"Positive score: {result['positive_score']:.2f}")
        print(f"Negative score: {result['negative_score']:.2f}")

if __name__ == "__main__":
    main() 