"""
Symphony AIDE - LLM Utilities for Extensions
Provides a unified interface to call open-source LLMs via HuggingFace Inference API
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("llm_router.llm_utils")

# Load .env file automatically
try:
    from dotenv import load_dotenv
    # Try to find .env in project root (one level up from core/)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info("Loaded .env from: %s", env_path)
    else:
        logger.warning(".env not found at %s", env_path)
except ImportError:
    logger.warning("python-dotenv not installed")
    pass  # python-dotenv not installed, rely on system env vars

# Try to import openai (for HuggingFace router)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Try to import google-genai
try:
    from google import genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# Fallback to requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LLMClient:
    """
    Client for calling open-source LLMs via HuggingFace Router API
    OR Google GenAI models via Google API.
    """
    
    # Models
    HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"
    GOOGLE_MODEL = "gemma-3-27b-it"
    
    # Default is still HF for now, unless specified
    DEFAULT_MODEL = GOOGLE_MODEL
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """Initialize LLM client."""
        self.model = model or self.DEFAULT_MODEL
        
        # Determine provider and key
        self.provider = "hf"
        if "gemma" in self.model or "gemini" in self.model:
            self.provider = "google"
            self.api_key = os.environ.get("New_GOOGLE_API_KEY")
            logger.info("Provider: Google, API key configured=%s", bool(self.api_key))
        else:
            self.provider = "hf"
            self.api_key = api_key or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
            logger.info("Provider: HuggingFace, API key configured=%s", bool(self.api_key))
        
        # Initialize Clients
        self._hf_client = None
        self._google_client = None
        
        if self.provider == "hf" and HAS_OPENAI and self.api_key:
            self._hf_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.api_key
            )
            logger.info("HuggingFace client initialized")
            
        elif self.provider == "google" and HAS_GOOGLE and self.api_key:
            # Temporary: Use hardcoded key until .env is updated with non-leaked key
            self._google_client = genai.Client(api_key=self.api_key)
            logger.info("Google client initialized")
        else:
            logger.warning("Client not initialized. HAS_GOOGLE=%s has_key=%s", HAS_GOOGLE, bool(self.api_key))
        
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        fallback_template: Optional[str] = None
    ) -> str:
        """Generate text using the configured LLM."""
        res = None
        try:
            if self.provider == "hf" and self._hf_client:
                res = self._call_hf_api(prompt, max_tokens, temperature)
            elif self.provider == "google" and self._google_client:
                res = self._call_google_api(prompt, max_tokens, temperature)
                
            if res:
                return res
        except Exception as e:
            logger.exception("API call failed (%s): %s", self.provider, e)
        
        if fallback_template:
            return fallback_template
        
        return f"# Generated code placeholder\n# Prompt: {prompt[:100]}..."
    
    def _call_hf_api(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Call HuggingFace Router API using OpenAI-compatible interface"""
        if self._hf_client:
            completion = self._hf_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        return None

    def _call_google_api(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Call Google GenAI API"""
        if self._google_client:
            response = self._google_client.models.generate_content(
                model=self.model,
                contents=prompt,
                #config={'temperature': temperature, 'max_output_tokens': max_tokens}
            )
            logger.debug("Google response received")
            return response.text
        return None


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        # Default to HF initially, unless GOOGLE_MODEL is forced
        # To use Google, one would instantiate LLMClient(model="gemma-3-27b-it")
        _llm_client = LLMClient()
    return _llm_client


def generate_code(prompt: str, language: str = "python", fallback: str = None) -> str:
    """
    Convenience function to generate code.
    
    Args:
        prompt: Description of code to generate
        language: Programming language
        fallback: Fallback code if LLM fails
        
    Returns:
        Generated code
    """
    full_prompt = f"""Generate {language} code for the following:

{prompt}

Only output the code, no explanations. Start with the code directly."""
    
    return get_llm_client().generate(full_prompt, fallback_template=fallback)


def test_api():
    """
    Test function to verify the LLM API is working.
    Run: python llm_utils.py
    """
    logger.info("%s", "=" * 50)
    logger.info("LLM API Test")
    logger.info("%s", "=" * 50)
    
    client = LLMClient()
    
    # Check API key
    if client.api_key:
        logger.info("API key configured=True")
    else:
        logger.error("API key not set. Set HF_API_KEY in .env file or environment variable")
        return False
    
    logger.info("Model: %s", client.model)
    logger.info("HuggingFace Hub: %s", "Available" if HAS_OPENAI else "Not installed (pip install openai)")
    
    # Test API call
    logger.info("Testing API call...")
    test_prompt = "Say hello in one word"
    
    try:
        result = client.generate(test_prompt, max_tokens=50, temperature=0.5)
        if result:
            logger.info("API response received")
            return True
        else:
            logger.error("API returned empty response")
            return False
    except Exception as e:
        logger.exception("API error: %s", e)
        return False


if __name__ == "__main__":
    test_api()




    '''
    [LLM] Loaded .env from: d:\VSCODE\IDE\.env
    ==================================================
    LLM API Test
    ==================================================
    [OK] API Key: hf_wAdNVmN...dxeo
    [OK] Model: mistralai/Mistral-7B-Instruct-v0.2:featherless-ai
    [OK] HuggingFace Hub: Available

    Testing API call...
    [OK] API Response received!
    ----------------------------------------
    Hello. That was one word. Is that what you meant? If so, mission accomplished. If not, please let me know and I'll do my best to help out.
    ----------------------------------------
    '''
