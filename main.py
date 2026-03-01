'''
Main application for dynamic LLM routing using LangGraph
'''
import os
import sys
import logging

# add core and config to path for imports
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SemanticCache
from core import Router
from config import *
from config.logger_config import setup_logger

# Model configuration
MODELS_CONFIG=MODELS_CONFIG
logger = logging.getLogger("llm_router.main")

def process_query(query: str, router: Router) -> None:
    """Process a single query using the router."""
    logger.info("%s", "=" * 50)
    logger.info("Processing query")
    
    # Route the query
    result = router.route(query)
    
    # Process the result
    if result.get("cache_hit"):
        logger.info("Cache hit")
    else:
        logger.info("Classification: %s", result.get("classification"))
        logger.info("Selected model: %s", result.get("selected_model"))
    
    if result.get("error"):
        logger.error("Error: %s", result["error"])
    
    if result.get("llm_response"):
        logger.info("Response from %s", result.get("used_model", "unknown model"))
        logger.info("%s", "-" * 50)
        logger.info("%s", result["llm_response"])
        logger.info("%s", "-" * 50)

def main():
    setup_logger("llm_router")
    # Initialize components
    cache = SemanticCache(default_ttl=600)  # 10 minute TTL
    classifier = Classifier()
    llm_client = LLMClient(MODELS_CONFIG)
    
    # Create the router
    router = Router(
        models_config={
            "tier1": [m[1] for m in MODELS_CONFIG["tier1"]],
            "tier2": [m[1] for m in MODELS_CONFIG["tier2"]],
            "tier3": [m[1] for m in MODELS_CONFIG["tier3"]],
        },
        cache=cache,
        classifier=classifier,
        llm_client=llm_client,
        max_retries=3
    )
    
    # Example queries
    queries = [
        "What is the capital of Ghana?",
        "Explain quantum computing in simple terms.",
        "Create code for a simple weather application that takes a city name and displays the current temperature by calling a weather API.",
        "Develop a multi-step plan to reduce carbon emissions in a mid-sized city, considering economic, social, and political factors."
    ]
    
    # Process each query
    for query in queries:
        process_query(query, router)


if __name__ == "__main__":
    main()
        