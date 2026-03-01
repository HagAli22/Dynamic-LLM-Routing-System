import os
import sys
import logging

# Add parent directory to path to import llm_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_utils import get_llm_client

logger = logging.getLogger("llm_router.classifier")


def classify_text(text):

    # define client
    client = get_llm_client()

    test_prompt = f"""You are a query classification engine.

    Your task is to classify the user's query into ONE of the following categories:

    S = Simple
    - Basic factual recall
    - Straightforward definition
    - Single-step question
    - No deep reasoning required
    - Can be handled by a low-cost model

    M = Medium
    - Summarization
    - Explanation with simplification
    - Moderate reasoning
    - Structured but not deeply multi-step
    - Some synthesis required

    A = Advanced
    - Multi-step reasoning
    - Strategic planning
    - Complex analysis
    - Coding tasks
    - Multi-intent queries
    - Deep logical or mathematical reasoning
    - Creative writing requiring high coherence
    - If a query contains multiple tasks, classify based on the MOST COMPLEX task.

    Rules:
    - Output ONLY one letter: S, M, or A.
    - Do NOT explain.
    - Do NOT answer the query.
    - If the query is ambiguous or missing context, classify as S.

    Examples:

    User: What is the capital of Japan?
    Output: S

    User: Summarize the following paragraph about mitosis: [text]
    Output: M

    User: Develop a multi-step plan to reduce carbon emissions in a mid-sized city.
    Output: A

    User: Tell me about that thing.
    Output: S

    User: Translate this
    Output: S

    User: How can I write a simple summary of a complex analysis?
    Output: M

    User: Who was the 16th US president and write a python script to calculate the length of his Gettysburg Address?
    Output: A

    User: Explain the theory of relativity to a five-year-old.
    Output: M

    User: Write a short creative story about a robot who discovers music.
    Output: A

    Now classify:

    User: {text}
    Output:
    """
        

    result = client.generate(test_prompt, max_tokens=50, temperature=0.5)
    if result and not result.startswith("# Generated code placeholder"):
        logger.debug("Classifier API response received")
        return result
    else:
        logger.warning("Classifier API returned empty response")
        return "No response"
    
