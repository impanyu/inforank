import json
from trust_framework import TrustFramework
from typing import List, Dict
import os
from datasets import load_dataset
import random
from data_utils import DataLoader

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
    dataset = load_dataset("wikipedia", "20231101.en", split="train")
    
    # Sample random articles
    samples = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    texts = []
    queries = []
    for idx in samples:
        article = dataset[idx]
        # Get all paragraphs from the article that are long enough
        paragraphs = [p for p in article['text'].split('\n') if len(p) > 100]
        if paragraphs:
            texts.extend(paragraphs)  # Add all paragraphs
            queries.append(article['title'])  # Add title for each paragraph
    
    return texts, queries

def main():
    # Initialize framework and data loader
    framework = TrustFramework(
        minimal_trustworthiness=0.2,
        consistency_threshold=0.5,
        credulity_factor=0.5,
        db_path="./test_db"
    )
    data_loader = DataLoader()
    
    try:
        # Load dataset and decompose into statements
        query_to_statements = data_loader.load_wikipedia(num_samples=1000)
        
        # Save to file for later use
        data_loader.save_to_json(query_to_statements, './data/processed_dataset.json')
        
        # Populate database with all statements
        all_statements = []
        for statements in query_to_statements.values():
            all_statements.extend(statements)
        
        print(f"\nPopulating database with {len(all_statements)} statements...")
        for i, statement in enumerate(all_statements, 1):
            result = framework.input(statement)
            print(f"Processed statement {i}/{len(all_statements)}")
        
        # Test queries
        print(f"\nTesting {len(query_to_statements)} queries...")
        for i, query in enumerate(query_to_statements.keys(), 1):
            print(f"\nQuery {i}: {query}")
            result = framework.retrieve(query)
            print(f"Retrieved: {result}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 