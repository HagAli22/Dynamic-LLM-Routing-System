import os
import time
import requests
import logging
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger("llm_router.fallback")

# Load env vars
load_dotenv()

# Map models to keys
MODEL_KEY_MAP = {
    "mistralai/mistral-7b-instruct:free": "mistral-7b-instruct",
    "qwen/qwen2.5-vl-32b-instruct:free": "qwen2.5-vl-32b-instruct",
    "meta-llama/llama-3.3-8b-instruct:free": "llama-3.3-8b-instruct",
    "qwen/qwen-2.5-72b-instruct:free": "qwen-2.5-72b-instruct",
    "openai/gpt-oss-20b:free": "gpt-oss-20b",
    "mistralai/devstral-small-2505:free": "devstral-small-2505",
    "qwen/qwq-32b:free": "qwq-32b",
    "qwen/qwen-2.5-coder-32b-instruct:free": "qwen-2.5-coder-32b-instruct",
    "deepseek/deepseek-r1-distill-llama-70b:free": "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-3.3-70b-instruct:free": "llama-3.3-70b-instruct",
}

# Price per 1M tokens
PRICES = {

    # prices for tier1
    "qwen/qwen-2.5-72b-instruct:free": {"input": 0.15, "output": 0.20},
    "qwen/qwen-2.5-coder-32b-instruct:free": {"input": 0.15, "output": 0.20},
    "meta-llama/llama-3.3-8b-instruct:free": {"input": 0.15, "output": 0.20},

    # prices for tier2
    "openai/gpt-oss-20b:free": {"input": 0.35, "output": 0.50},
    "mistralai/devstral-small-2505:free": {"input": 0.35, "output": 0.50},
    "mistralai/mistral-7b-instruct:free": {"input": 0.35, "output": 0.50},
    
    # prices for tier3
    "qwen/qwen-2.5-coder-32b-instruct:free": {"input": 0.70, "output": 1.00},
    "deepseek/deepseek-r1-distill-llama-70b:free": {"input": 0.70, "output": 1.00},
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.70, "output": 1.00},
}

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count based on word count."""
    words = text.split()
    return int(len(words) * 1.3)  # Assume 1.3 tokens per word on average

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calc cost based on token usage"""
    if model_name not in PRICES:
        return 0.0
    price = PRICES[model_name]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


class FallbackChatGradientAI:
    """Try multiple models until success"""

    def __init__(self, models: list[str]):
        self.models = models

    def invoke(self, prompt: str, max_retries: int = 3):
        """Send prompt, fallback if model fails"""
        last_exception = None

        for model_name in self.models:
            logger.info("Attempting request with model: %s", model_name)
            model_name=model_name
            if isinstance(model_name, (list, tuple)) and len(model_name) > 1:
                logger.debug("Selected model tuple: %s", model_name)
                model_name = model_name[1]  # Use the model identifier
            logger.debug("Resolved model_name=%s", model_name)
            api_key_name = MODEL_KEY_MAP.get(model_name)
            logger.debug("Resolved api_key_name=%s", api_key_name)
            api_key = os.getenv(api_key_name)

            if not api_key:
                logger.warning("API key for model '%s' not found. Skipping this model.", model_name)
                continue  # Skip if no key
            
            client = genai.Client(api_key=api_key)
            logger.debug("Client initialized for model=%s", model_name)
            

            

            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    response = client.models.generate_content(
                        model="gemma-3-27b-it",
                        contents=prompt,
                        config={
                            "temperature": 0.7,
                            "max_output_tokens": 1024,
                        }
                    )
                    end_time = time.time()

                    result = response.text

                    usage = response.usage_metadata
                    
                    input_tokens = usage.prompt_token_count or 0

                    output_tokens = estimate_tokens(result)

                    cost = calculate_cost(model_name, input_tokens, output_tokens)
                    end_time = time.time()

                    return {
                        "model": model_name,
                        "response": result,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": cost,
                        "time_taken": end_time - start_time,
                    }

                    

                except Exception as e:
                    last_exception = e
                    logger.exception("Model %s failed on attempt %s/%s", model_name, attempt + 1, max_retries)

        raise Exception(f"All models failed. Last error: {str(last_exception)}")
