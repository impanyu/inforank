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
from config import OPENAI_API_KEY  # Add this import at the top
from collections import deque
import os


@dataclass
class StoredItem:
    text: str
    vector: np.ndarray
    trust_score: float
    positive_score: float
    negative_score: float
    timestamp: str
    vector_id: int

class TrustFramework:
    def __init__(
        self,
        minimal_trustworthiness: float = 0.2,
        consistency_threshold: float = 0.5,
        credulity_factor: float = 0.5,
        vector_dim: int = 384,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        db_path: str = "./db",  # Add default database path
        search_k: int = 50,
        converge_max_iterations: int = 10,
        converge_threshold: float = 0.01,
        max_vectors_to_converge: int = 200,
        retrieve_minimal_trustworthiness: float = 0.2,
        retrieve_credulity_factor: float = 0.5,
        retrieve_consistency_threshold: float = 0.5,
        retrieve_search_k: int = 50,
        word_limit: int = 1000,
    ):
        """
        Initialize the Trust Framework with configurable parameters
        
        Args:
            minimal_trustworthiness: Minimum trust score required (0-1)
            consistency_threshold: Threshold for consistency checking (0-1)
            credulity_factor: Factor to weight new information (0-1)
            vector_dim: Dimension of vector embeddings
            api_key: OpenAI API key (optional, will use config if not provided)
            model: Model to use (default: gpt-4o)
        """
        self.minimal_trustworthiness = minimal_trustworthiness
        self.consistency_threshold = consistency_threshold
        self.credulity_factor = credulity_factor
        self.vector_dim = vector_dim
        self.search_k = search_k
        self.converge_max_iterations = converge_max_iterations
        self.converge_threshold = converge_threshold
        self.max_vectors_to_converge = max_vectors_to_converge
        self.retrieve_minimal_trustworthiness = retrieve_minimal_trustworthiness
        self.retrieve_credulity_factor = retrieve_credulity_factor
        self.retrieve_consistency_threshold = retrieve_consistency_threshold
        self.retrieve_search_k = retrieve_search_k
        self.word_limit = word_limit
        
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
        
        
        # Initialize LLM handler with config API key as fallback
        self.llm = LLMHandler(api_key=api_key or OPENAI_API_KEY, model=model)
        
        # Set database paths
        self.index_path = f"{db_path}/faiss_index.bin"
        self.metadata_path = f"{db_path}/metadata.json"
        
        # Try to load existing database
        if not self.load_db():
            # Initialize new database if loading fails
            self.stored_items = {}
            self.next_id = 0
        
    def text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector using the embedding model"""
        return self.embedder.encode([text])[0]
    
    def implication_check(self, text_1: str, text_2: str) -> str:
        prompt = LLMPrompts.IMPLICATION_CHECK.format(text_1=text_1, text_2=text_2)
        response = self.llm.generate(prompt)
        return response
    
    def find_implying_combinations(self, text: str, indices: List[int], positive_combinations: List[str], negative_combinations: List[str], current_combinations: List[str], start_index: int) -> list:
        if len(current_combinations) > 0:
            concatenated_text = " ".join([self.stored_items[index].text for index in current_combinations])
            response = self.implication_check(concatenated_text, text)

            if response == "Positive":
    
                for positive_combination in positive_combinations[:]:
                    comb_concatenated_text = " ".join([self.stored_items[index].text for index in positive_combination])
                    comb_response = self.implication_check(comb_concatenated_text, concatenated_text)
                    if comb_response == "Positive":
                        positive_combinations.remove(positive_combination)
                    
                    comb_reverse_response = self.implication_check(concatenated_text, comb_concatenated_text)
                    if comb_reverse_response == "Positive":
                        return             
                
                positive_combinations.append(current_combinations)
 
                return
            elif response == "Negative":
                for negative_combination in negative_combinations[:]:
                    comb_concatenated_text = " ".join([self.stored_items[index].text for index in negative_combination])
                    comb_response = self.implication_check(comb_concatenated_text, concatenated_text)
                    if comb_response == "Positive":
                        negative_combinations.remove(negative_combination)
                    
                    comb_reverse_response = self.implication_check(concatenated_text, comb_concatenated_text)
                    if comb_reverse_response == "Positive":
                        return             
                
                negative_combinations.append(current_combinations)

                return


        # backtracking to find all possible combinations
        for i in range(start_index, len(indices)):
            current_combinations.append(indices[i]) 
            self.find_implying_combinations(text, indices, positive_combinations, negative_combinations, current_combinations, i+1)
            current_combinations.pop()
        
        
    def calculate_scores(self, text: str) -> list:
        """Calculate positive and negative scores for this vector"""
        prompt = LLMPrompts.TRUST_EVALUATION.format(text=text)
        response = self.llm.generate(prompt)

        vector = self.text_to_vector(text)

        # Search for similar vectors
        k = min(self.search_k, self.index.ntotal)  # Get top-k results
        distances, indices = self.index.search(vector.reshape(1, -1), k)
        positive_combinations = []
        negative_combinations = []
        current_combinations = []

        # recursively calculate positive and negrative scores using combination of similar texts
        self.find_implying_combinations(text, indices[0], positive_combinations, negative_combinations, current_combinations, 0)
        positive_score = 0
        for combination in positive_combinations:
            product = 1
            for index in combination:
                product *= self.stored_items[index].trust_score
            positive_score += product

        negative_score = 0
        for combination in negative_combinations:
            product = 1
            for index in combination:
                product *= self.stored_items[index].trust_score
            negative_score += product
        
        return positive_score, negative_score
    
    def converge_trust_scores(self, vector_id: int):
        """first figure out the set of vectors need to be converged"""
        # define a set to store the vectors need to be converged
        vectors_to_converge = set()
        # define a queue to store the vectors 
        queue = deque()
        # add the vector_id to the queue
        queue.append(vector_id)

        while len(queue) > 0:
            # get the vector_id from the queue
            vector_id = queue.popleft()
            vectors_to_converge.add(vector_id)
            if len(vectors_to_converge) > self.max_vectors_to_converge:
                break
        
            # get the similar vectors
            distances, indices = self.index.search(self.stored_items[vector_id].vector.reshape(1, -1), self.search_k)
            for index in indices[0]:
                if index not in vectors_to_converge:
                    queue.append(index)
                

        # iteratively converge the trust scores
        for _ in range(self.converge_max_iterations):
            for vector_id in vectors_to_converge:
                positive_score, negative_score  = self.calculate_scores(self.stored_items[vector_id].text)
                trust_score = positive_score - negative_score + self.credulity_factor
                trust_score = min(trust_score, 1)
                trust_score = max(trust_score, 0)
                self.stored_items[vector_id].trust_score = trust_score
                self.stored_items[vector_id].positive_score = positive_score
                self.stored_items[vector_id].negative_score = negative_score


    


    def handle_long_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Handle text that exceeds the word limit by decomposing it
        
        Args:
            text: Input text to be processed
            
        Returns:
            List of Dictionary containing processed information and trust metrics
        """
        # Decompose the text
        prompt = LLMPrompts.DECOMPOSE.format(text=text)
        response = self.llm.generate(prompt)
        
        if response == "False":
            # Even long text cannot be meaningfully decomposed
            return []
            
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
            return []

    def input(self, text: str) -> List[Dict[str, Any]]:
        """
        Process and store new information with trust metrics
        
        Args:
            text: Input text to be processed
            
        Returns:
            List of Dictionary containing processed information and trust metrics
        """
        # Check for long text first
        if self.count_words(text) > self.word_limit:
            return self.handle_long_text(text)

        # Calculate vector first to avoid duplicate calculations
        vector = self.text_to_vector(text)
        
        # Calculate scores
        positive_score, negative_score = self.calculate_scores(vector)

        consistency_score = positive_score/(abs(negative_score)+0.01)
        
        trust_score = positive_score - negative_score + self.credulity_factor
        trust_score = min(trust_score, 1)
        trust_score = max(trust_score, 0)


        # Check trustworthiness and consistency
        if trust_score >= self.minimal_trustworthiness and consistency_score >= self.consistency_threshold:
            # Add to FAISS index and get the vector_id
            vector_id = self.next_id  # Store ID before adding
            self.index.add(vector.reshape(1, -1))
            
        
            
            # Store metadata with the correct vector_id
            stored_item = StoredItem(
                text=text,
                vector=vector,
                trust_score=trust_score,
                positive_score=positive_score,
                negative_score=negative_score,
                timestamp=datetime.now().isoformat(),
                vector_id=vector_id  # Use the stored ID
            )
            self.stored_items[vector_id] = stored_item
            self.next_id += 1  # Increment ID after successful addition
            

            self.converge_trust_scores(vector_id)
            return [{
                'text': text,
                'trust_score': self.stored_items[vector_id].trust_score,
                'positive_score': self.stored_items[vector_id].positive_score,
                'negative_score': self.stored_items[vector_id].negative_score,
                'vector_id': vector_id,
                'timestamp': datetime.now().isoformat()
            }]
        
        else:
            return self.handle_long_text(text)

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant information pieces based on query
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary of relevant information piece with trust scores
        """
        if self.index.ntotal == 0:
            return {'text': '', 'trust_score': 0, 'positive_score': 0, 'negative_score': 0}
            
        # Convert query to vector
        query_vector = self.text_to_vector(query)
        
        # Search for similar vectors
        k = min(self.retrieve_search_k, self.index.ntotal)  # Get top-k results
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        
        # Collect results above minimal trustworthiness
        results = []
        for idx in indices[0]:
            stored_item = self.stored_items[int(idx)]  
            results.append({
                'text': stored_item.text,
                'trust_score': stored_item.trust_score,
                'positive_score': stored_item.positive_score,
                'negative_score': stored_item.negative_score,
                'vector_id': stored_item.vector_id,
                'timestamp': stored_item.timestamp
            })
        
        # Sort by trust score
        results.sort(key=lambda x: x['trust_score'], reverse=True)
        positive_results = []
        negative_results = []
        trust_score = 0
        positive_score = 1
        negative_score = 0
        result_text = ''

        for i in range(len(results)):
            result = results[i]
            if result in negative_results:
                continue

            new_positive_results = positive_results[:]
            new_negative_results = negative_results[:]
            new_trust_score = 0
            new_positive_score = 1
            new_negative_score = 0
            new_result_text = ''

            for j in range(len(positive_results)-1, -1, -1):
                positive_result = positive_results[j]
                response = self.implication_check(result['text'], positive_result['text'])
                if response == "Positive":
                    new_positive_score /= (positive_result['trust_score']+0.01)
                    break
  
            new_positive_score *= result['trust_score']
            new_positive_results.append(result)
            new_result_text += " " + result['text']

            for k in range(i+1, len(results)):
                response = self.implication_check(result['text'], results[k]['text'])
                
                if response == "Negative":
                    add_negative = True
                    for negative_result in new_negative_results[:]:
                        response = self.implication_check(results[k]['text'], negative_result['text'])
                        if response == "Positive":
                            add_negative = False
                            break
                        
                        response = self.implication_check(negative_result['text'], results[k]['text'])
                        if response == "Positive":
                            new_negative_results.remove(negative_result)
                    if add_negative:
                        new_negative_score += results[k]['trust_score']
                        new_negative_results.append(results[k])

            new_trust_score =  self.retrieve_credulity_factor + new_positive_score - new_negative_score
            new_trust_score = min(new_trust_score, 1)
            new_trust_score = max(new_trust_score, 0)
           
            new_consistency_score = new_positive_score/(abs(new_negative_score)+0.01)

            if new_trust_score >= self.retrieve_minimal_trustworthiness and new_consistency_score >= self.retrieve_consistency_threshold:
                positive_results = new_positive_results
                negative_results = new_negative_results
                result_text = new_result_text
                trust_score = new_trust_score
                positive_score = new_positive_score
                negative_score = new_negative_score
            else:
                break
                
        
        return {'text': result_text, 'trust_score': trust_score, 'positive_score': positive_score, 'negative_score': negative_score}

    def remove_vector(self, vector_id: int):
        """Remove a vector from the index"""
        if hasattr(self.index, 'remove_ids'):
            self.index.remove_ids(np.array([vector_id]))
            del self.stored_items[vector_id]


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

    def calculate_l2_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate L2 (Euclidean) distance between two vectors
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: L2 distance between the vectors
        """
        # Ensure vectors are the same shape
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector shapes do not match: {vector1.shape} != {vector2.shape}")
            
        # Calculate L2 distance using numpy
        return float(np.linalg.norm(vector1 - vector2)) 

    def clear_all(self):
        """
        Clear all vectors and associated data from the framework
        """
        # Reset FAISS index
        self.index.reset()  # Clear all vectors from FAISS index
        
        # Clear stored items
        self.stored_items.clear()
        
        # Reset ID counter
        self.next_id = 0 

    def count_words(self, text: str) -> int:
        """
        Count the number of words in a string
        
        Args:
            text: Input text string
            
        Returns:
            int: Number of words in the text
        """
        # Remove extra whitespace and split
        words = text.strip().split()
        return len(words) 

    def write_db(self) -> bool:
        """
        Save the FAISS index and metadata to disk using member paths
        
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Prepare metadata for serialization
            serializable_items = {}
            for id, item in self.stored_items.items():
                serializable_items[id] = {
                    'text': item.text,
                    'vector': item.vector.tolist(),
                    'trust_score': item.trust_score,
                    'positive_score': item.positive_score,
                    'negative_score': item.negative_score,
                    'timestamp': item.timestamp,
                    'vector_id': item.vector_id
                }
            
            metadata = {
                'stored_items': serializable_items,
                'next_id': self.next_id
            }
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            return True
                
        except Exception as e:
            print(f"Error saving database: {e}")
            return False

    def load_db(self) -> bool:
        """
        Load the FAISS index and metadata from disk using member paths
        
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            # Check if both files exist
            if not (os.path.exists(self.index_path) and os.path.exists(self.metadata_path)):
                print("Database files not found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Restore stored items
            self.stored_items = {}
            for id_str, item_dict in metadata['stored_items'].items():
                id = int(id_str)
                self.stored_items[id] = StoredItem(
                    text=item_dict['text'],
                    vector=np.array(item_dict['vector']),
                    trust_score=item_dict['trust_score'],
                    positive_score=item_dict['positive_score'],
                    negative_score=item_dict['negative_score'],
                    timestamp=item_dict['timestamp'],
                    vector_id=item_dict['vector_id']
                )
            
            # Restore next_id
            self.next_id = metadata['next_id']
            
            return True
            
        except Exception as e:
            print(f"Error loading database: {e}")
            return False 