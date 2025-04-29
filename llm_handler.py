from typing import Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from config import OPENAI_API_KEY

class LLMHandler:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize LLM handler
        
        Args:
            api_key: OpenAI API key (optional, will use config if not provided)
            model: Model to use (default: gpt-4o)
        """
        self.client = openai.OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that responds exactly as instructed."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for more consistent outputs
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            raise 