from typing import List, Tuple, Dict
from datasets import load_dataset
import random
import json
import os
from llm_handler import LLMHandler
from llm_prompts import LLMPrompts
from config import OPENAI_API_KEY

class DataLoader:
    def __init__(self):
        """Initialize DataLoader with LLM handler"""
        self.llm = LLMHandler(api_key=OPENAI_API_KEY)

    def decompose_text(self, text: str) -> List[str]:
        """
        Decompose text into independent statements using LLM
        
        Args:
            text: Text to decompose
            
        Returns:
            List of independent statements
        """
        prompt = LLMPrompts.DECOMPOSE.format(text=text)
        response = self.llm.generate(prompt)
        
        try:
            if response == "False":
                # Text cannot be decomposed further
                return [text]
            statements = json.loads(response)
            return statements
        except json.JSONDecodeError:
            return [text]

    def load_wikipedia(self, num_samples: int = 100) -> Dict[str, List[str]]:
        """
        Load Wikipedia dataset and decompose articles into statements
        
        Args:
            num_samples: Number of articles to sample
            
        Returns:
            Dict mapping queries to lists of statements
        """
        print("Loading Wikipedia dataset...")
        dataset = load_dataset("wikipedia", "20231101.en", split="train")
        
        samples = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        query_to_statements = {}
        for i, idx in enumerate(samples):
            article = dataset[idx]
            title = article['title']
            
            # Directly decompose the full article text
            statements = self.decompose_text(article['text'])
            
            if statements:
                query_to_statements[title] = statements
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} articles")
        
        return query_to_statements

    def load_ms_marco(self, num_samples: int = 100) -> Dict[str, List[str]]:
        """
        Load MS MARCO dataset and decompose passages into statements
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            Dict mapping queries to lists of statements
        """
        print("Loading MS MARCO dataset...")
        dataset = load_dataset("ms_marco", "v1.1")
        
        train_data = dataset['train']
        samples = random.sample(range(len(train_data)), min(num_samples, len(train_data)))
        
        query_to_statements = {}
        for i, idx in enumerate(samples):
            example = train_data[idx]
            if example['query']:
                query = example['query']
                passages = example.get('passages', {})
                
                # Get parallel arrays
                is_selected = passages.get('is_selected', [])
                passage_texts = passages.get('passage_text', [])
                
                # Only decompose selected passages
                statements = []
                for selected, text in zip(is_selected, passage_texts):
                    if selected == 1 and text:
                        statements.extend(self.decompose_text(text))
                
                if statements:
                    query_to_statements[query] = statements
                    
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")
        
        return query_to_statements

    def load_natural_questions(self, num_samples: int = 100) -> Dict[str, List[str]]:
        """
        Load Natural Questions dataset and decompose documents into statements
        
        Returns:
            Dict mapping queries to lists of statements
        """
        print("Loading Natural Questions dataset...")
        dataset = load_dataset("natural_questions")
        
        samples = random.sample(range(len(dataset['test'])), min(num_samples, len(dataset['test'])))
        
        query_to_statements = {}
        for i, idx in enumerate(samples):
            example = dataset['test'][idx]
            if example['document']['html'] and example['question']['text']:
                question = example['question']['text']
                
                # decompose each
                statements = self.decompose_text(example['document']['html'])
               
                if statements:
                    query_to_statements[question] = statements
                    
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} examples")
        
        return query_to_statements

    @staticmethod
    def save_to_json(query_to_statements: Dict[str, List[str]], output_path: str):
        """Save query-to-statements mapping to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(query_to_statements, f, indent=2)

    @staticmethod
    def load_from_json(input_path: str) -> Dict[str, List[str]]:
        """Load query-to-statements mapping from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text by cleaning and normalizing
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\t', ' ').replace('\r', '')
        
        # Basic normalization
        text = text.strip()
        
        return text

    @staticmethod
    def filter_texts(texts: List[str], min_length: int = 100, max_length: int = 1000) -> List[str]:
        """
        Filter texts based on length and content
        
        Args:
            texts: List of texts to filter
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered list of texts
        """
        filtered = []
        for text in texts:
            text = DataLoader.preprocess_text(text)
            if min_length <= len(text) <= max_length:
                filtered.append(text)
        return filtered

    @staticmethod
    def save_to_json(texts: List[str], queries: List[str], output_dir: str):
        """
        Save texts and queries to JSON files
        
        Args:
            texts: List of texts
            queries: List of queries
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'texts.json'), 'w') as f:
            json.dump({'texts': texts}, f, indent=2)
            
        with open(os.path.join(output_dir, 'queries.json'), 'w') as f:
            json.dump({'queries': queries}, f, indent=2) 