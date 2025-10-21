import asyncio
from typing import Dict, List, Literal,Optional
import os
import sys
from langgraph.graph import StateGraph,END,MessagesState
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# add core and config to path for imports
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import SemanticCache , classify_text , FallbackChatGradientAI
from core import Router
from config import *

# Model configuration
MODELS_CONFIG=MODELS_CONFIG

# Disable LangSmith if no API key is provided
if not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("LangSmith tracing disabled - no API key provided")

# Define the full state
class RouterState(MessagesState):

    query: Optional[str] = None
    classification: Optional[Literal["S", "M", "A"]] = None
    model_tier: Optional[Literal["tier1", "tier2", "tier3"]] = None
    selected_model: Optional[str] = None
    cache_hit: bool = False
    cached_response: Optional[str] = None
    llm_response: Optional[str] = None
    used_model: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0


# Helper function to get latest human message
def get_latest_human_message(messages: List[BaseMessage]) -> str:

    if not messages:
        return ""
    
    # Get the latest human message (iterate from the end)
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            
            # Handle different content formats from Chat vs Graph interface
            if isinstance(content, list):
                # Chat interface sends: [{'type': 'text', 'text': 'actual message'}]
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        return item.get('text', '')
            elif isinstance(content, str):
                # Graph interface sends: "actual message"
                return content
            else:
                return str(content)
    
    return ""

# Router Logic - Synchronous versions for Chat compatibility
class StudioRouter:
    def __init__(self, models_config, cache, classifier, llm_client=None, max_retries=3):
        self.models_config = models_config
        self.cache = cache
        self.classifier = classifier
        self.llm_client = llm_client
        self.max_retries = max_retries

    def store_in_cache(self, state: RouterState) -> RouterState:
        """Store response in cache - synchronous version"""
        try:
            if state.get("llm_response") and not state.get("cache_hit", False):
                query = state.get("query", "")
                if not query:
                    query = get_latest_human_message(state.get("messages", []))
                
                response = state["llm_response"]
                self.cache.set(query, response)
                print(f"[DEBUG] Stored in cache - Query: '{query}' | Response: '{response[:50]}...'")
        except Exception as e:
            print(f"[WARNING] Failed to store in cache: {e}")
        return state

    def check_cache(self, state: RouterState) -> RouterState:

        try:
            # Always extract fresh query from messages
            messages = state.get("messages", [])
            query = get_latest_human_message(messages)
            
            if not query:
                print(f"[WARNING] No query found in messages")
                return {**state, "cache_hit": False, "query": ""}

            response = self.cache.get(query)
            if response is not None:
                print(f"Cache HIT - Found response for: '{query}'")
                new_messages = list(messages) + [AIMessage(content=response)]
                return {
                    **state,
                    "cache_hit": True,
                    "cached_response": response,
                    "llm_response": response,
                    "messages": new_messages,
                    "query": query,
                }
            
            print(f"Cache MISS - No response found for: '{query}'")
            return {**state, "cache_hit": False, "query": query}
        except Exception as e:
            print(f"ERROR Cache check failed: {e}")
            query = get_latest_human_message(state.get("messages", []))
            return {**state, "cache_hit": False, "error": str(e), "query": query}

    def classify_query(self, state: RouterState) -> RouterState:

        try:
            # Always get fresh query from messages, don't rely on state["query"]
            messages = state.get("messages", [])
            query = get_latest_human_message(messages)
                        
            if not query:
                return {**state, "error": "No query found for classification"}
            
            classification = self.classifier.classify_text(query)
            if classification not in ["S", "M", "A"]:
                return {**state, "error": f"Invalid classification: {classification}"}
            
            return {**state, "classification": classification, "query": query}
        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            return {**state, "error": f"Error in classify_query: {e}"}

    def select_model(self, state: RouterState) -> RouterState:

        try:
            tier = state.get("classification")
            if not tier:
                return {**state, "error": "No classification available"}
            
            retry_count = state.get("retry_count", 0)
            tier_key = f"tier{1 if tier=='S' else 2 if tier=='M' else 3}"
            models = self.models_config.get(tier_key, [])
            
            if not models:
                return {**state, "error": f"No models available for {tier_key}"}
            
            selected_model = models[retry_count % len(models)]
            
            return {
                **state, 
                "model_tier": tier_key, 
                "selected_model": selected_model, 
                "retry_count": retry_count + 1
            }
        except Exception as e:
            print(f"[ERROR] Model selection failed: {e}")
            return {**state, "error": f"Error in select_model: {e}"}

    def call_llm(self, state: RouterState) -> RouterState:

        try:
            if not state.get("selected_model"):
                return {**state, "error": "No model selected"}
            
            # Always get fresh query from messages
            messages = state.get("messages", [])
            query = get_latest_human_message(messages)

            if not query:
                return {**state, "error": "No query found for LLM call"}

            if self.llm_client:

                result = self.llm_client.call_sync(
                    model=state["selected_model"], 
                    messages=messages,
                    query=query,
                    tier=state["model_tier"]
                )
                
                if isinstance(result, dict):
                    response = result.get("response") or result.get("output_text") or str(result)
                else:
                    response = str(result)
            else:
                response = f"Mock response for: {query}"

            new_messages = list(messages) + [AIMessage(content=response)]
                        
            return {
                **state,
                "llm_response": response,
                "used_model": state["selected_model"],
                "messages": new_messages,
                "query": query,
            }
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            return {**state, "error": str(e)}

    def handle_error(self, state: RouterState) -> RouterState:
        error = state.get("error", "Unknown error")
        print(f"[ERROR] Handling error: {error}")
        return state

    def should_use_cache(self, state: RouterState) -> str:
        
        return "use_cache" if state.get("cache_hit") else "classify"

    def handle_llm_response(self, state: RouterState) -> str:

        if state.get("llm_response"):
            return "success"
        return "error"

    def should_retry(self, state: RouterState) -> str:

        retry_count = state.get("retry_count", 0)
        max_retries = self.max_retries
        
        if retry_count < max_retries and not state.get("llm_response"):
            print(f"Retrying attempt {retry_count}/{max_retries}")
            return "retry"
        return "fail"
    

# Classifier + LLM Client
class Classifier:
    def __init__(self):
        self.classify_text = classify_text

class LLMClient:
    def __init__(self, models_config: Dict[str, List[str]]):
        self.models_config = models_config
        self.fallback_handlers = {
            tier: FallbackChatGradientAI(models=models_list)
            for tier, models_list in models_config.items()
        }

    def call_sync(self, model: str, messages: List[BaseMessage], query: str ,tier) -> str:
        """Synchronous version of call method for Chat interface"""
        # Use the explicitly passed query, or extract from messages
        if query:
            actual_query = query
        else:
            actual_query = get_latest_human_message(messages)
        
        # Pick the correct fallback handler (default = tier1 if not found)
        print(f"[DEBUG] Using tier: '{tier}' for model: ''")
        fallback = self.fallback_handlers.get(tier, self.fallback_handlers["tier1"])

        result = fallback.invoke(actual_query)
        
        return result

    async def call(self, model: str, messages: List[BaseMessage], query: str = None) -> str:
        """Async version for backward compatibility"""
        return self.call_sync(model, messages, query)


cache = SemanticCache(default_ttl=600)
classifier = Classifier()
llm_client = LLMClient({k:[m[1] for m in v] for k,v in MODELS_CONFIG.items()})
router = StudioRouter(
    models_config={k:[m[1] for m in v] for k,v in MODELS_CONFIG.items()}, 
    cache=cache, 
    classifier=classifier, 
    llm_client=llm_client
)


# Build Graph
def create_graph():
    builder = StateGraph(RouterState)
    
    # Add nodes - all synchronous now
    builder.add_node("check_cache", router.check_cache)
    builder.add_node("classify_query", router.classify_query)
    builder.add_node("select_model", router.select_model)
    builder.add_node("call_llm", router.call_llm)
    builder.add_node("handle_error", router.handle_error)
    builder.add_node("store_in_cache", router.store_in_cache)

    # Set entry point
    builder.set_entry_point("check_cache")
    
    # Add edges
    builder.add_conditional_edges(
        "check_cache", 
        router.should_use_cache, 
        {"use_cache": END, "classify": "classify_query"}
    )
    builder.add_edge("classify_query", "select_model")
    builder.add_edge("select_model", "call_llm")
    builder.add_conditional_edges(
        "call_llm", 
        router.handle_llm_response, 
        {"success": "store_in_cache", "error": "handle_error"}
    )
    builder.add_edge("store_in_cache", END)
    builder.add_conditional_edges(
        "handle_error", 
        router.should_retry, 
        {"retry": "select_model", "fail": END}
    )
    
    return builder.compile()


graph = create_graph()

# Sample input for Studio
sample_inputs = [
    {"messages": [HumanMessage(content="What is the capital of Egypt?")]},
    {"messages": [HumanMessage(content="Hello, how are you?")]},
]





