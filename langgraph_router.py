import logging
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

MAX_RETRIES_PER_MODEL = 3
WORKFLOW_RECURSION_LIMIT = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouterState(TypedDict):
    """The state of the router workflow."""


    query: str


    classification: Optional[Literal["S", "M", "A"]]


    model_tier: Optional[Literal["tier1", "tier2", "tier3"]]
    selected_model: Optional[str]


    cache_hit: bool
    cached_response: Optional[str]


    llm_response: Optional[str]
    used_model: Optional[str]


    error: Optional[str]
    retry_count: int


class Router:
    """
    LangGraph-based router for handling query classification,
    model selection, caching, and retries.
    """

    def __init__(
        self,
        models_config: dict,
        cache,
        classifier,
        llm_client=None,
        max_retries: int = MAX_RETRIES_PER_MODEL,
    ):
        self.models_config = models_config
        self.cache = cache
        self.classifier = classifier
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.workflow = self._create_workflow()


    def store_in_cache(self, state: RouterState) -> RouterState:
        """Store the LLM response in the semantic cache if not already cached."""
        try:
            if state.get("llm_response") and not state.get("cache_hit", False):
                query = state["query"]
                response = state["llm_response"]
                if isinstance(response, dict) and "response" in response:
                    response_text = response["response"]
                else:
                    response_text = str(response)

                self.cache.set(query, response_text)
                logger.debug(f"Stored response in semantic cache for query: {query[:50]}...")
            return state
        except Exception as e:
            logger.warning(f"Failed to store in cache: {str(e)}")
            return state

    def check_cache(self, state: RouterState) -> RouterState:
        """Check if the query already exists in the semantic cache."""
        logger.debug("Checking cache for query")
        print("DEBUG: Cache check state:")
        response = self.cache.get(state["query"])
        if response is not None:
            return {
                **state,
                "cache_hit": True,
                "cached_response": response,
                "llm_response": response,
            }
        return {**state, "cache_hit": False}


    def classify_query(self, state: RouterState) -> RouterState:
        """Classify the query into Small (S), Medium (M), or Advanced (A)."""
        logger.debug("Classifying query")
        print("DEBUG: Classification state:")
        try:
            if not state.get("query"):
                state["error"] = "No query provided for classification"
                return state

            classification = self.classifier.classify_text(state["query"])

            if classification not in ["S", "M", "A"]:
                state["error"] = f"Invalid classification: {classification}"
                return state

            logger.debug(f"Query classified as: {classification}")
            print(f"[DEBUG] Query classified as: {classification}")
            return {**state, "classification": classification, "error": None}

        except Exception as e:
            state["error"] = f"Error in classify_query: {str(e)}"
            logger.error(state["error"])
            return state


    def select_model(self, state: RouterState) -> RouterState:
        """Select a model from the tier based on classification and retry count."""
        logger.debug("Selecting model")
        print("DEBUG: Model selection state:")
        state = dict(state)

        try:
            tier = state.get("classification")
            if not tier:
                state["error"] = "No classification available"
                return state

            tier_key = self._map_classification_to_tier(tier)
            print(f"[DEBUG] Mapped classification {tier} to tier key {tier_key}")
            models = self.models_config.get(tier_key, [])

            if not models:
                state["error"] = f"No models available for {tier_key}"
                return state

            retry_count = state.get("retry_count", 0)
            max_retries = len(models) * MAX_RETRIES_PER_MODEL

            if retry_count >= max_retries:
                state["error"] = f"Max retries ({max_retries}) exceeded for {tier_key}"
                return state

            model_idx = retry_count % len(models)
            selected_model = models[model_idx]

            if isinstance(selected_model, (list, tuple)) and len(selected_model) > 1:
                selected_model = selected_model[1]

            state.update(
                {
                    "model_tier": tier_key,
                    "selected_model": selected_model,
                    "retry_count": retry_count + 1,
                    "error": None,
                }
            )

            logger.info(
                f"Selected model: {selected_model} for tier {tier_key} (attempt {retry_count + 1}/{max_retries})"
            )
            print(f"[INFO] Selected model: {selected_model} for tier {tier_key} (attempt {retry_count + 1}/{max_retries})")
            return state

        except Exception as e:
            state["error"] = f"Error in select_model: {str(e)}"
            logger.error(state["error"])
            return state

    def _map_classification_to_tier(self, classification: str) -> str:
        """Helper to map classification letter to tier key."""
        mapping = {"S": "tier1", "M": "tier2", "A": "tier3"}
        return mapping.get(classification, "tier1")


    def call_llm(self, state: RouterState) -> RouterState:
        """Call the LLM client with the selected model."""
        logger.debug("Calling LLM")
        print("DEBUG: LLM call state:")
        if not state.get("selected_model"):
            return {**state, "error": "No model selected for LLM call"}

        try:
            model_identifier = state["selected_model"]
            print(f"[DEBUG] Using model identifier: {model_identifier}")
            if isinstance(model_identifier, (list, tuple)) and len(model_identifier) > 1:
                model_identifier = model_identifier[1]

            if self.llm_client:
                response = self.llm_client.call(
                    model_identifier,
                    [{"role": "user", "content": state["query"]}],
                    state["model_tier"],
                )
            else:
                logger.warning("No LLM client provided, using mock response")
                response = f"Response for: {state['query']} from {model_identifier}"

            self.cache.set(state["query"], response)

            return {**state, "llm_response": response, "used_model": state["selected_model"]}

        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return {**state, "error": str(e)}


    def handle_error(self, state: RouterState) -> RouterState:
        """Pass through state after error."""
        return state

    def should_use_cache(self, state: RouterState) -> str:
        """Decide whether to use cache or classify query."""
        return "use_cache" if state.get("cache_hit") else "classify"

    def handle_llm_response(self, state: RouterState) -> str:
        """Decide next step after LLM response."""
        print("DEBUG: LLM response handler state:")
        if state.get("llm_response"):
            return "success"

        if state.get("error"):
            return "error"

        retry_count = state.get("retry_count", 0)
        tier = state.get("classification", "S")
        max_models = len(self.models_config.get(self._map_classification_to_tier(tier), []))
        max_retries = max_models * MAX_RETRIES_PER_MODEL

        if retry_count >= max_retries:
            state["error"] = f"Max retries ({max_retries}) exceeded"
            return "error"

        return "retry"

    def should_retry(self, state: RouterState) -> str:
        """Decide whether to retry or fail after error."""
        retry_count = state.get("retry_count", 0)
        tier = state.get("classification", "S")
        max_models = len(self.models_config.get(self._map_classification_to_tier(tier), []))
        max_retries = max_models * MAX_RETRIES_PER_MODEL

        return "retry" if retry_count < max_retries else "fail"


    def _create_workflow(self):
        """Build LangGraph workflow with nodes and edges."""
        builder = StateGraph(RouterState)

        # Define nodes
        builder.add_node("check_cache", self.check_cache)
        builder.add_node("classify_query", self.classify_query)
        builder.add_node("select_model", self.select_model)
        builder.add_node("call_llm", self.call_llm)
        builder.add_node("handle_error", self.handle_error)
        builder.add_node("store_in_cache", self.store_in_cache)

        # Define edges
        builder.set_entry_point("check_cache")
        builder.add_conditional_edges("check_cache", self.should_use_cache, {"use_cache": END, "classify": "classify_query"})
        builder.add_edge("classify_query", "select_model")
        builder.add_edge("select_model", "call_llm")
        builder.add_conditional_edges("call_llm", self.handle_llm_response, {"success": "store_in_cache", "retry": "select_model", "error": "handle_error"})
        builder.add_edge("store_in_cache", END)
        builder.add_conditional_edges("handle_error", self.should_retry, {"retry": "select_model", "fail": END})

        # Stop condition
        def should_stop(state: RouterState) -> bool:
            return bool(state.get("llm_response") or state.get("cached_response") or state.get("error"))

        workflow = builder.compile()
        workflow.recursion_limit = WORKFLOW_RECURSION_LIMIT
        workflow.should_stop = should_stop

        return workflow


    def route(self, query: str) -> RouterState:
        """Run a query through the workflow and return the final state."""
        initial_state = RouterState(
            query=query,
            classification=None,
            model_tier=None,
            selected_model=None,
            cache_hit=False,
            cached_response=None,
            llm_response=None,
            used_model=None,
            error=None,
            retry_count=0,
        )
        result = self.workflow.invoke(initial_state)
        return result
