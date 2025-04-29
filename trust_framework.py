from typing import Dict, List, Any
import numpy as np
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Optional
from llm_prompts import LLMPrompts
import json
from llm_handler import LLMHandler

@dataclass
class StoredItem:
    text: str
    trust_score: float
    consistency_score: float
    timestamp: str
    vector_id: int

class TrustFramework:
    def __init__(
        self,
        minimal_trustworthiness: float = 0.2,
        consistency_threshold: float = 1.0,
        credulity_factor: float = 0.5,
        vector_dim: int = 384,
        api_key: Optional[str] = None,
        model: str = "gpt-4o"
    ):
        """
        Initialize the Trust Framework with configurable parameters
        
        Args:
            minimal_trustworthiness: Minimum trust score required (0-1)
            consistency_threshold: Threshold for consistency checking (0-1)
            credulity_factor: Factor to weight new information (0-1)
            vector_dim: Dimension of vector embeddings
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o)
        """
        self.minimal_trustworthiness = minimal_trustworthiness
        self.consistency_threshold = consistency_threshold
        self.credulity_factor = credulity_factor
        self.vector_dim = vector_dim
        
        
        # Initialize HNSW index - very fast search with good accuracy
        self.index = faiss.IndexHNSWFlat(vector_dim, 32)  # 32 is M (max connections)
        self.index.hnsw.efConstruction = 200  # Higher value = better accuracy but slower construction
        self.index.hnsw.efSearch = 40  # Higher value = better accuracy but slower search
        
        # Optional: Add GPU support
        if faiss.get_num_gpus():
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Store for text and metadata
        self.stored_items = {}  # id -> StoredItem
        self.next_id = 0
        
        # Additional array to store trust scores aligned with vectors
        self.trust_scores = []
        
        # Initialize LLM handler
        self.llm = LLMHandler(api_key=api_key, model=model)
        
    def text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector using the embedding model"""
        return self.embedder.encode([text])[0]
        
    def calculate_trust_score(self, text: str) -> float:
        """Calculate trust score based on various factors"""
        prompt = LLMPrompts.TRUST_EVALUATION.format(text=text)
        response = self.llm.generate(prompt)
        try:
            return float(response)
        except ValueError:
            return 0.0  # Default to 0 if response is not a valid float

    def calculate_consistency(self, text: str) -> float:
        """
        Calculate consistency with existing information using vector similarity
        """
        if self.index.ntotal == 0:
            return 1.0
            
        query_vector = self.text_to_vector(text)
        
        # Find most similar vectors
        k = min(5, self.index.ntotal)  # Get top-k similar items
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Convert L2 distance to similarity score (closer to 1 means more similar)
        similarities = 1 / (1 + distances[0])
        
        # Weight similarities by trust scores of retrieved items
        weighted_sim = 0
        total_weight = 0
        for idx, sim in zip(indices[0], similarities):
            trust_score = self.trust_scores[idx]
            weighted_sim += sim * trust_score
            total_weight += trust_score
            
        return weighted_sim / total_weight if total_weight > 0 else 0.8

    def input(self, text: str) -> list:
        """
        Process and store new information with trust metrics
        
        Args:
            text: Input text to be processed
            
        Returns:
            Dictionary containing processed information and trust metrics
        """
        vector = self.text_to_vector(text)


        trust_score = self.calculate_trust_score(text)
        consistency_score = self.calculate_consistency(text)
        
        # Check trustworthiness and consistency
        # if condition is met, add text to the index
        if trust_score >= self.minimal_trustworthiness and consistency_score <= self.consistency_threshold:
            # Convert text to vector
            
            
            # Add to FAISS index
            self.index.add(vector.reshape(1, -1))
            
            # Store trust score
            self.trust_scores.append(trust_score)  # Store trust_score 
            
            # Store metadata
            stored_item = StoredItem(
                text=text,
                trust_score=trust_score,
                consistency_score=consistency_score,
                timestamp=datetime.now().isoformat(),
                vector_id=self.next_id
            )
            self.stored_items[self.next_id] = stored_item
            self.next_id += 1
        # the text is decomposable
        else:
            # Decompose the text
            prompt = LLMPrompts.DECOMPOSE.format(text=text)
            # Send to LLM and get response
            response = self.llm.generate(prompt)  # This is a placeholder - implement actual LLM call
            
            if response == "False":
                # Text cannot be decomposed further, ignore the text
                return []
            else:
                try:
                    # Parse JSON list of decomposed units
                    decomposed_units = json.loads(response)
                    
                    # Process each decomposed unit recursively
                    results = []
                    for unit in decomposed_units:
                        results.extend(self.input(unit))
                    return results
                    
                except json.JSONDecodeError:
                    # Handle invalid JSON response
                    return [{
                        'text': text,
                        'trust_score': trust_score,
                        'consistency_score': consistency_score,
                        'timestamp': datetime.now().isoformat()
                    }]

    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant information pieces based on query
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant information pieces with trust scores
        """
        if self.index.ntotal == 0:
            return []
            
        # Convert query to vector
        query_vector = self.text_to_vector(query)
        
        # Search for similar vectors
        k = min(10, self.index.ntotal)  # Get top-k results
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Collect results above minimal trustworthiness
        results = []
        for idx in indices[0]:
            stored_item = self.stored_items[int(idx)]
            if stored_item.trust_score >= self.minimal_trustworthiness:  # Use trust_score instead
                results.append({
                    'text': stored_item.text,
                    'trust_score': stored_item.trust_score,
                    'consistency_score': stored_item.consistency_score,
                    'timestamp': stored_item.timestamp
                })
        
        # Sort by trust score
        results.sort(key=lambda x: x['trust_score'], reverse=True)
        
        return results 

    def remove_vector(self, vector_id: int):
        """Remove a vector from the index"""
        if hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(np.array([vector_id]))
            del self.stored_items[vector_id]
            # Update trust scores array
            del self.trust_scores[vector_id]

    def batch_add(self, texts: List[str]):
        """Add multiple vectors efficiently"""
        vectors = self.embedder.encode(texts)
        self.index.add(vectors)
        # ... store metadata ...

    def train_index(self, training_vectors: np.ndarray):
        """Train the index for better organization of vectors"""
        if hasattr(self.index, 'train'):
            self.index.train(training_vectors)

    def configure_index(self, 
        index_type: str = 'ivf_flat',
        vector_dim: int = 384,
        nlist: int = 100,  # for IVF
        m: int = 32,  # for HNSW
        ef_construction: int = 200,  # for HNSW
        ef_search: int = 40,  # for HNSW
        use_gpu: bool = False
    ) -> faiss.Index:
        """
        Configure FAISS index with specified parameters
        
        Args:
            index_type: Type of index ('flat', 'ivf_flat', 'ivf_pq', 'hnsw', 'lsh')
            vector_dim: Dimension of vectors
            nlist: Number of clusters for IVF indexes
            m: Number of connections for HNSW
            ef_construction: Construction time/accuracy trade-off for HNSW
            ef_search: Search time/accuracy trade-off for HNSW
            use_gpu: Whether to use GPU
        """
        if index_type == 'flat':
            index = faiss.IndexFlatL2(vector_dim)
        
        elif index_type == 'ivf_flat':
            quantizer = faiss.IndexFlatL2(vector_dim)
            index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist)
            index.nprobe = 10  # Number of clusters to visit during search
        
        elif index_type == 'ivf_pq':
            quantizer = faiss.IndexFlatL2(vector_dim)
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, bits)
            index.nprobe = 10
        
        elif index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(vector_dim, m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        
        elif index_type == 'lsh':
            nbits = 32
            index = faiss.IndexLSH(vector_dim, nbits)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Move to GPU if requested and available
        if use_gpu and faiss.get_num_gpus():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index 