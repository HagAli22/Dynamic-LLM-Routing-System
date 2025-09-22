import os
import sys
from typing import Dict, List
from enum import Enum
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from classifier import classify_text
from fallback import FallbackChatGradientAI
from semantic_cache import SemanticCache

MODELS_CONFIG = {
    "tier1": [
        ["llama-3.3-8b-instruct", "meta-llama/llama-3.3-8b-instruct:free"],
        ["qwen-2.5-72b-instruct", "qwen/qwen-2.5-72b-instruct:free"],
        ["qwq-32b", "qwen/qwq-32b:free"],
        
    ],
    "tier2": [
        ["gpt-oss-20b", "openai/gpt-oss-20b:free"],
        ["mistral-7b-instruct", "mistralai/mistral-7b-instruct:free"],        
        ["devstral-small-2505", "mistralai/devstral-small-2505:free"],
    ],
    "tier3": [
        ["qwen-2.5-coder-32b-instruct", "qwen/qwen-2.5-coder-32b-instruct:free"],
        ["deepseek-r1-distill-llama-70b", "deepseek/deepseek-r1-distill-llama-70b:free"],
        ["llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free"],
    ]
}

class Classifier:
    """Wrapper for the classifier to match the expected interface."""
    
    def __init__(self):
        self.classify_text = classify_text

class LLMClient:
    """Wrapper for the LLM client to match the expected interface."""
    
    def __init__(self, models_config: Dict[str, List[List[str]]]):
        self.models_config = models_config
        # Initialize fallback handlers for each tier
        self.fallback_handlers = {
            tier: FallbackChatGradientAI(models=models_list)
            for tier, models_list in models_config.items()
        }
    
    def call(self, model: str, messages: List[Dict[str, str]], tier: str) -> str:
        """Call the LLM with the given messages and force correct tier fallback."""
        # Extract the query (assume one user message)
        query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

        # Pick the correct fallback handler (default = tier1 if not found)
        fallback = self.fallback_handlers.get(tier, self.fallback_handlers["tier1"])

        # Call the model
        response = fallback.invoke(query)

        # Return clean response
        return response[0] if isinstance(response, (list, tuple)) else response


