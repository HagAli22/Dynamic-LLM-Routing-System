'''
Main application for dynamic LLM routing using LangGraph
'''
from semantic_cache import SemanticCache
from langgraph_router import Router
from config import *

# Model configuration
MODELS_CONFIG=MODELS_CONFIG

def process_query(query: str, router: Router) -> None:
    """Process a single query using the router."""
    print(f"\n{'='*50}")
    print(f"Processing query: {query}")
    
    # Route the query
    result = router.route(query)
    
    # Process the result
    if result.get("cache_hit"):
        print("Cache hit! Using cached response.")
    else:
        print(f"Classification: {result.get('classification')}")
        print(f"Selected model: {result.get('selected_model')}")
    
    if result.get("error"):
        print(f"Error: {result['error']}")
    
    if result.get("llm_response"):
        print(f"\nResponse from {result.get('used_model', 'unknown model')}:")
        print("-" * 50)
        print(result["llm_response"])
        print("-" * 50)

def main():
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
        