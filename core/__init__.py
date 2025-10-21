"""
user-defined core modules for the Dynamic LLM Routing System.
"""
import sys
from pathlib import Path

# add core to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

from fallback import FallbackChatGradientAI
from semantic_cache import SemanticCache
from langgraph_router import Router
from classifier import classify_text

__all__ = [
    "FallbackChatGradientAI",
    "SemanticCache",
    "Router"
    "classify_text"
]