"""
Streamlit App for Dynamic LLM Routing using LangGraph
"""

import streamlit as st
import time
import pandas as pd
from io import StringIO
from config import *

# Import modules
try:
    from semantic_cache import SemanticCache
    from langgraph_router import Router
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Model configuration (tiers)
MODELS_CONFIG =MODELS_CONFIG

class SimpleCache:
    """Basic cache fallback"""
    def __init__(self):
        self.cache = {}
    def get(self, key):
        return self.cache.get(key)
    def set(self, key, value, ttl=None):
        self.cache[key] = value


@st.cache_resource
def initialize_router():
    """Init router with cache"""
    try:
        cache = SemanticCache(default_ttl=600)
        st.success("✅ Semantic cache initialized")
    except Exception as e:
        st.warning(f"⚠️ Semantic cache failed, using simple cache: {e}")
        cache = SimpleCache()

    classifier = Classifier()
    llm_client = LLMClient(MODELS_CONFIG)

    return Router(
        models_config={k: [m[1] for m in v] for k, v in MODELS_CONFIG.items()},
        cache=cache,
        classifier=classifier,
        llm_client=llm_client,
        max_retries=3
    )


def process_single_query(query: str, router: Router):
    """Process one query through router"""
    start_time = time.time()
    try:
        result = router.route(query)
        cache_hit = result.get("cache_hit", False)
        classification = result.get("classification", "Unknown")
        model_tier = result.get("model_tier", "tier1")
        selected_model = result.get("selected_model", "Unknown")

        messages = [{"role": "user", "content": query}]
        llm_response = router.llm_client.call(selected_model, messages, model_tier)
        response = result.get("llm_response") or llm_response or result.get("cached_response", "")
        if isinstance(response, dict):
            actual_response = response.get("response", str(response))
        else:
            actual_response = str(response) if response else ""

        return {
            "success": True,
            "cache_hit": cache_hit,
            "classification": classification,
            "model_tier": model_tier,
            "selected_model": selected_model,
            "used_model": result.get("used_model", "Unknown"),
            "response": actual_response,
            "error": result.get("error"),
            "speed": time.time() - start_time,
            "raw_result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "speed": time.time() - start_time,
            "cache_hit": False,
            "classification": "Error",
            "response": f"Error: {str(e)}"
        }


def calculate_accuracy(query, response):
    """Word overlap accuracy"""
    if not query or not response:
        return 0.0
    q_words = set(str(query).lower().split())
    r_words = set(str(response).lower().split())
    return len(q_words & r_words) / len(q_words) if q_words else 0.0


def main():
    st.set_page_config(page_title="LangGraph LLM Router", page_icon="🚀", layout="wide")
    st.title("🚀 Dynamic LLM Routing with LangGraph")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    mode = st.sidebar.radio("Select Mode:", ["🔍 Single Query", "🧪 Batch Testing"])

    with st.spinner("🔧 Initializing router..."):
        router = initialize_router()

    # Sidebar model info
    with st.sidebar.expander("📋 Model Tiers Info"):
        for tier in ["tier1", "tier2", "tier3"]:
            st.write(f"**{tier.capitalize()}:** {len(MODELS_CONFIG[tier])} models")

    if mode == "🔍 Single Query":
        st.header("Single Query Processing")
        query = st.text_area("Enter your query:", height=100, placeholder="Type your question...")
        if st.button("🚀 Process Query", type="primary", disabled=not query.strip()):
            with st.spinner("🔄 Processing query..."):
                result = process_single_query(query, router)
            if result["success"]:
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⚡ Speed", f"{result['speed']:.2f}s")
                col2.metric("💾 Cache", "Hit" if result["cache_hit"] else "Miss")
                col3.metric("🎯 Classification", result["classification"])
                col4.metric("📊 Accuracy", f"{calculate_accuracy(query, result['response'])*100:.1f}%")

                # Response
                st.subheader("💬 Response")
                if result["error"]:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.text_area("Response:", result["response"], height=300, disabled=True)
            else:
                st.error(f"❌ Query failed: {result['error']}")

    elif mode == "🧪 Batch Testing":
        st.header("Batch Testing Suite")
        input_method = st.radio("Choose input method:", ["📝 Manual Entry", "📁 Upload File", "🎲 Predefined"])
        test_queries = []

        if input_method == "📝 Manual Entry":
            q_text = st.text_area("Enter test queries:", height=200)
            if q_text:
                test_queries = [q.strip() for q in q_text.split("\n") if q.strip()]
        elif input_method == "📁 Upload File":
            file = st.file_uploader("Upload text file", type=['txt'])
            if file:
                content = StringIO(file.getvalue().decode("utf-8")).read()
                test_queries = [q.strip() for q in content.split('\n') if q.strip()]
        else:
            predefined = [
                "Who wrote Hamlet?",
                "What is the capital of France?",
                "Explain machine learning in simple terms.",
                "How do solar panels work?"
            ]
            test_queries = st.multiselect("Select predefined queries:", predefined, default=predefined[:3])

        if test_queries and st.button("🚀 Run Batch Tests", type="primary"):
            progress = st.progress(0)
            results = []
            for i, query in enumerate(test_queries):
                progress.progress((i+1)/len(test_queries))
                result = process_single_query(query, router)
                results.append({
                    "Query": query,
                    "Response": result.get("response", ""),
                    "Classification": result.get("classification", "Unknown"),
                    "Model_Tier": result.get("model_tier", "Unknown"),
                    "Used_Model": result.get("used_model", "Unknown"),
                    "Speed_s": round(result.get("speed", 0), 2),
                    "Accuracy": round(calculate_accuracy(query, result.get("response", ""))*100, 1),
                    "Cache": "Hit" if result.get("cache_hit") else "Miss",
                    "Error": result.get("error", "")
                })
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")


if __name__ == "__main__":
    main()
