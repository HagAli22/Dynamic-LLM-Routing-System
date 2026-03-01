import os
import sys
from uuid import uuid4

# add core and config to path for imports
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from core import SemanticCache
from core import Router
from core import classify_text
from core import FallbackChatGradientAI
from config import *
from config.logger_config import setup_test_logger
import time


logger = setup_test_logger()

# from config.config import MODELS_CONFIG

class TestSuite:
    def __init__(self,MODELS_CONFIG):
        self.logger = logger
        self.active_query_id = "-"
        self.cache = SemanticCache()

        # Define models for different tiers

        self.models = MODELS_CONFIG


        self.results = []
        self.logger.info(
            "TestSuite initialized with tiers=%s",
            {tier: len(models) for tier, models in self.models.items()},
        )



    def _check_cache(self, query):
        self.logger.info("[qid=%s] Checking semantic cache", self.active_query_id)
        return self.cache.get(query)

    def _is_cache_hit(self, cached_response):
        return cached_response is not None

    def _get_cache_result(self, cached_response):
        return {
            "response": cached_response,
            "route": "cache",
            "used_model": "cache",
            "speed": 0.1,
            "cost": 0.0
        }

    def _classify_query(self, query):
        classification = classify_text(query)
        classification = classification.strip().upper()
        self.logger.info("[qid=%s] Classification result: %s", self.active_query_id, classification)
        if isinstance(classification, dict):
            return classification.get("route", "tier1")
        else:
            return str(classification) if classification else "tier1"

    def _get_models_for_route(self, route):
        mapping = {"S": "tier1", "M": "tier2", "A": "tier3"}
        tier = mapping.get(route, "tier1")
        return [m[1] for m in self.models.get(tier, [])]

    def _get_fallback_models(self):
        models_for_route = []
        for tier in ("tier1", "tier2", "tier3"):
            models_for_route.extend([m[1] for m in self.models.get(tier, [])])
        return models_for_route

    def _invoke_model(self, query, models_list):
        self.logger.info("[qid=%s] Invoking fallback with %s candidate models", self.active_query_id, len(models_list))
        fallback = FallbackChatGradientAI(models=models_list)
        return fallback.invoke(query)

    def _parse_model_response(self, model_response, models_list, start_time):
        if isinstance(model_response, dict):
            self.logger.info(
                "[qid=%s] Model response received (model=%s, cost=%.6f)",
                self.active_query_id,
                model_response.get("model", ""),
                model_response.get("cost", 0.0),
            )
            return {
                "response": model_response.get("response", ""),
                "used_model": model_response.get("model", ""),
                "speed": model_response.get("time", time.time() - start_time),
                "tokens": model_response.get("tokens", 0),
                "cost": model_response.get("cost", 0.0)
            }
        else:
            self.logger.info("[qid=%s] Non-dict response received from model fallback", self.active_query_id)
            return {
                "response": str(model_response),
                "used_model": models_list[0] if models_list else "unknown",
                "speed": time.time() - start_time,
                "tokens": 0,
                "cost": 0.0
            }

    def _handle_model_error(self, error, start_time):
        self.logger.exception("Error getting response: %s", str(error))
        return {
            "response": f"Error: {str(error)}",
            "used_model": "error",
            "speed": time.time() - start_time,
            "cost": 0.0
        }


    def _invoke_model(self, query, models_list):
        fallback = FallbackChatGradientAI(models=models_list)
        return fallback.invoke(query)

    def _parse_model_response(self, model_response, models_list, start_time):
        if isinstance(model_response, dict):
            return {
                "response": model_response.get("response", ""),
                "used_model": model_response.get("model", ""),
                "speed": model_response.get("time", time.time() - start_time),
                "tokens": model_response.get("tokens", 0),
                "cost": model_response.get("cost", 0.0)
            }
        else:
            return {
                "response": str(model_response),
                "used_model": models_list[0] if models_list else "unknown",
                "speed": time.time() - start_time,
                "tokens": 0,
                "cost": 0.0
            }

    def _handle_model_error(self, error, start_time):
        self.logger.exception("Error getting response: %s", str(error))
        return {
            "response": f"Error: {str(error)}",
            "used_model": "error",
            "speed": time.time() - start_time,
            "cost": 0.0
        }

    def _cache_response(self, query, response):
        self.cache.set(query, response)
        self.logger.info("[qid=%s] Response cached", self.active_query_id)


    def _create_result_entry(self, query, response_data, route, cache_hit):
        return {
            "Query": query,
            "Response": response_data["response"],
            "Route": route,
            "UsedModel": response_data["used_model"],
            "Speed": round(response_data["speed"], 2),
            "Cost": f"${response_data['cost']:.6f}",
            "Cache": "Hit" if cache_hit else "Miss"
        }

    def _save_result(self, result_entry):
        self.results.append(result_entry)
        self.logger.info(
            "[qid=%s] Result saved (route=%s, model=%s, cache=%s, speed=%.2fs)",
            self.active_query_id,
            result_entry["Route"],
            result_entry["UsedModel"],
            result_entry["Cache"],
            result_entry["Speed"],
        )


    def _handle_cache_flow(self, query):
        cached_response = self._check_cache(query)
        cache_hit = self._is_cache_hit(cached_response)

        if cache_hit:
            self.logger.info("[qid=%s] Cache hit", self.active_query_id)
            return self._get_cache_result(cached_response), cache_hit

        self.logger.info("[qid=%s] Cache miss", self.active_query_id)
        return None, cache_hit

    def _determine_models_to_use(self, query):
        route = self._classify_query(query)
        models_for_route = self._get_models_for_route(route)
        self.logger.info("[qid=%s] Route selected: %s", self.active_query_id, route)

        # If no models found for the classified route, fall back to all models
        if not models_for_route:
            self.logger.warning("[qid=%s] No models for route=%s, using global fallback list", self.active_query_id, route)
            models_for_route = self._get_fallback_models()

        self.logger.info("[qid=%s] Candidate models prepared: %s", self.active_query_id, len(models_for_route))
        return route, models_for_route


    def _execute_model_invocation(self, query, models_list, start_time):
        try:
            model_response = self._invoke_model(query, models_list)
            response_data = self._parse_model_response(model_response, models_list, start_time)
            self.logger.info(
                "[qid=%s] Model invocation done (used_model=%s, speed=%.2fs)",
                self.active_query_id,
                response_data.get("used_model", "unknown"),
                response_data.get("speed", 0.0),
            )
            return response_data

        except Exception as e:
            return self._handle_model_error(e, start_time)

    def _handle_model_flow(self, query, start_time):
        route, models_list = self._determine_models_to_use(query)
        response_data = self._execute_model_invocation(query, models_list, start_time)

        # Cache the response
        self._cache_response(query, response_data["response"])

        return response_data, route

    def get_results_table(self):
        """Return results as pandas DataFrame"""
        return pd.DataFrame(self.results)

    def run_test(self, query, expected_route=None):
        self.active_query_id = uuid4().hex[:8]
        self.logger.info("[qid=%s] Starting test for query: %s", self.active_query_id, query)

        start_time = time.time()

        # Try cache flow first
        cache_result, cache_hit = self._handle_cache_flow(query)

        if cache_result:
            # Cache hit
            response_data = cache_result
            route = cache_result["route"]
        else:
            # Cache miss
            response_data, route = self._handle_model_flow(query, start_time)

        # Create and save result entry
        result_entry = self._create_result_entry(query, response_data, route, cache_hit)
        self._save_result(result_entry)
        self.logger.info(
            "[qid=%s] Finished test (error=%s)",
            self.active_query_id,
            str(response_data.get("used_model", "")) == "error",
        )


if __name__ == "__main__":

    MODELS_CONFIG= MODELS_CONFIG
    test_suite = TestSuite(MODELS_CONFIG)

    test_queries = [
        "Give me a one-sentence explanation of photosynthesis.",
        "Who wrote the play 'Hamlet'?",
        "Translate 'Thank you' into Japanese.",
        "Draft a formal apology email to a client for a missed deadline.",
        "Give me a one-sentence explanation of photosynthesis.",
        "Explain quantum computing as if I am 10 years old.",
        "Create a Java class for managing a library system with books and members.",
        "Who wrote the play 'Hamlet'?",
        "Refactor this Python function to improve readability and performance: def f(a): return [x for x in a if x%2==0]",
        "Design a REST API endpoint (URL, HTTP method, request/response JSON) for user authentication using JWT, include example request and response.",
        "Explain the difference between a process and a thread in operating systems in 5 concise bullet points.",
        "Summarize the causes of the American Civil War in 6 bullet points.",
        "Refactor this Python function to improve readability and performance: def f(a): return [x for x in a if x%2==0]",
        "Draft a formal apology email to a client for a missed deadline.",
        "Explain the difference between a process and a thread in operating systems in 5 concise bullet points.",
        "Create a Java class for managing a library system with books and members."
    ]


    for query in test_queries:
        logger.info("Processing query: %s", query)
        test_suite.run_test(query)

    # Get results
    results_df = test_suite.get_results_table()

    display_df = results_df.copy()
    display_df["Response"] = display_df["Response"].apply(lambda s: (s[:200] + "...") if isinstance(s, str) and len(s) > 200 else s)
    logger.info("=== Test Results ===")
    logger.info("\n%s", display_df[["Query", "Route", "UsedModel", "Speed", "Cost", "Cache"]])


    excel_file = "test_results.xlsx"
    results_df.to_excel(excel_file, index=False)
    logger.info("Results have been saved to: %s", excel_file)


    logger.info("=== Summary Statistics ===")
    logger.info("Total Queries: %s", len(results_df))
    logger.info("Cache Hits: %s", len(results_df[results_df['Cache'] == 'Hit']))
    logger.info("Average Speed: %.2f seconds", results_df['Speed'].mean())
    total_cost = sum(float(cost.strip('$')) for cost in results_df['Cost'])
    logger.info("Total Cost: $%.6f", total_cost)
