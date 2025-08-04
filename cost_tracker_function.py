"""
title: Live Cost Tracker when Chatting
description: Manages and calculates costs for model usage in the Chat
authors: brammittendorff
author_url: https://github.com/brammittendorff/openwebui-pipelines
funding_url: https://github.com/open-webui
version: 0.0.2
license: MIT
requirements: requests, tiktoken, cachetools, pydantic
environment_variables:
disclaimer: Provided as-is without warranties.
            You must ensure it meets your needs.
"""

import hashlib
import json
import os
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from threading import Lock
from typing import Any, Awaitable, Callable, Optional

import requests
import tiktoken
from cachetools import TTLCache, cached
from pydantic import BaseModel, Field


class Config:
    DATA_DIR = "data"
    CACHE_DIR = os.path.join(DATA_DIR, ".cache")
    USER_COST_FILE = os.path.join(DATA_DIR, f"costs-{datetime.now().year}.json")

    # Use the new remote JSON with model pricing
    REMOTE_JSON_URL = (
        "https://raw.githubusercontent.com/"
        "/rbb-dev/openwebui-pipelines/refs/heads/main/json/model_pricing.json"
    )

    CACHE_TTL = 432000  # e.g., 5 days
    CACHE_MAXSIZE = 8
    DECIMALS = "0.00000001"
    DEBUG = False


def debug_print(msg: str):
    if Config.DEBUG:
        print("[COST TRACKER DEBUG] " + msg)


def get_encoding_for_model(model_name: str):
    """
    Safely get a tiktoken encoding for the given model_name,
    falling back to 'cl100k_base' if unknown.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        debug_print(f"Unknown encoding for model={model_name}, using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")


class UserCostManager:
    def __init__(self, cost_file_path: str):
        self.cost_file_path = cost_file_path
        self._ensure_cost_file_exists()

    def _ensure_cost_file_exists(self):
        if not os.path.exists(self.cost_file_path):
            with open(self.cost_file_path, "w", encoding="UTF-8") as f:
                json.dump({}, f)

    def _read_costs(self) -> dict:
        with open(self.cost_file_path, "r", encoding="UTF-8") as f:
            return json.load(f)

    def _write_costs(self, costs: dict):
        with open(self.cost_file_path, "w", encoding="UTF-8") as f:
            json.dump(costs, f, indent=2)

    def update_user_cost(
        self,
        user_email: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_cost: Decimal,
    ):
        """
        Append a cost record for the given user & model to the JSON file.
        """
        costs_data = self._read_costs()
        if user_email not in costs_data:
            costs_data[user_email] = []

        costs_data[user_email].append(
            {
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": str(total_cost),
            }
        )
        self._write_costs(costs_data)


cache = TTLCache(maxsize=Config.CACHE_MAXSIZE, ttl=Config.CACHE_TTL)


class ModelCostManager:
    _lock = Lock()

    def __init__(self, remote_url: str, fallback_dict: dict):
        self.remote_url = remote_url
        self.fallback_dict = fallback_dict  # now empty
        self.cached_data = None  # Will store the remote JSON data (if downloaded)
        self.cache_file = os.path.join(Config.CACHE_DIR, "model_prices.json")
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    @cached(cache=cache)
    def fetch_remote_data(self) -> dict:
        """
        Attempt to download remote JSON. If that fails, return empty dict.
        """
        if not self.remote_url:
            return {}
        try:
            debug_print(f"Attempting to fetch remote cost data from {self.remote_url}")
            resp = requests.get(self.remote_url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            debug_print(f"Failed to fetch remote data: {e}")
            return {}

    def get_model_data(self, model: str) -> dict:
        """
        Return the cost data for a specific model with improved matching.
        1) Try exact match in remote data
        2) Try partial match if exact match fails
        3) If not found, fallback to empty => zero cost
        """
        if not self.cached_data:
            with self._lock:
                if not self.cached_data:
                    self.cached_data = self.fetch_remote_data()
                    debug_print(
                        f"Loaded model pricing data with {len(self.cached_data)} models"
                    )
                    debug_print(
                        f"Available models: {list(self.cached_data.keys())[:5]}..."
                    )

        # First try exact match
        if self.cached_data and model in self.cached_data:
            debug_print(f"Found exact match for model '{model}'")
            return self.cached_data[model]

        # If exact match fails, try to find a partial match
        # This handles cases where model names differ slightly
        if self.cached_data:
            # Try to match the model name without version numbers or prefixes
            base_model = model.split("-")[0] if "-" in model else model
            for key in self.cached_data.keys():
                if base_model in key:
                    debug_print(f"Found partial match for '{model}' using '{key}'")
                    return self.cached_data[key]

        # Otherwise fallback to empty => zero cost
        debug_print(
            f"No cost data found for '{model}' in remote data. Using zero cost."
        )
        return {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }


class CostCalculator:
    def __init__(
        self, user_cost_manager: UserCostManager, model_cost_manager: ModelCostManager
    ):
        self.user_cost_manager = user_cost_manager
        self.model_cost_manager = model_cost_manager

    def calculate_costs(
        self, model: str, input_tokens: int, output_tokens: int, compensation: float
    ) -> Decimal:
        """
        Calculate the cost of input_tokens + output_tokens for the given model.
        Use 'compensation' as a multiplier (e.g. 1.2 for 20% markup).
        """
        cost_data = self.model_cost_manager.get_model_data(model)
        in_cpt = Decimal(str(cost_data.get("input_cost_per_token", 0)))
        out_cpt = Decimal(str(cost_data.get("output_cost_per_token", 0)))

        input_cost = Decimal(input_tokens) * in_cpt
        output_cost = Decimal(output_tokens) * out_cpt
        raw_total = input_cost + output_cost
        final_cost = raw_total * Decimal(compensation)

        # Round to the nearest DECIMALS
        return final_cost.quantize(Decimal(Config.DECIMALS), rounding=ROUND_HALF_UP)


class Filter:
    class Valves(BaseModel):
        compensation: float = Field(default=1.0, description="Price multiplier")
        show_elapsed_time: bool = True
        show_tokens: bool = True
        show_tokens_per_second: bool = True
        debug: bool = False

    def __init__(self):
        self.valves = self.Valves()
        Config.DEBUG = self.valves.debug

        self.user_cost_manager = UserCostManager(Config.USER_COST_FILE)
        # fallback_dict is now empty => only remote used
        self.model_cost_manager = ModelCostManager(
            remote_url=Config.REMOTE_JSON_URL, fallback_dict={}  # <--- now empty
        )
        self.cost_calculator = CostCalculator(
            self.user_cost_manager, self.model_cost_manager
        )

        self.input_tokens = 0
        self.start_time = None

    def _remove_roles(self, text: str) -> str:
        """
        Remove lines that begin with 'SYSTEM:', 'USER:', 'ASSISTANT:', or 'PROMPT:'.
        """
        roles = ("SYSTEM:", "USER:", "ASSISTANT:", "PROMPT:")
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if any(line.startswith(r) for r in roles):
                cleaned.append(line.split(":", 1)[1].strip())
            else:
                cleaned.append(line)
        return "\n".join(cleaned).strip()

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Called before the main generation step:
         - Count input tokens
         - Possibly store user email
         - Mark start_time
        """
        messages = body.get("messages", [])
        content_str = "\n".join([m.get("content", "") for m in messages])
        cleaned_text = self._remove_roles(content_str)

        enc = get_encoding_for_model(body.get("model", "unknown-model"))
        self.input_tokens = len(enc.encode(cleaned_text))

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Input tokens: {self.input_tokens}",
                        "done": False,
                    },
                }
            )

        # If there's user info
        if __user__ and "email" in __user__:
            body["user"] = __user__["email"]

        self.start_time = time.time()
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Called after the generation step:
         - Count output tokens
         - Compute cost
         - Save cost
         - Emit stats
        """
        end_time = time.time()
        elapsed = end_time - self.start_time

        messages = body.get("messages", [])
        last_msg_content = messages[-1].get("content", "") if messages else ""
        enc = get_encoding_for_model(body.get("model", "unknown-model"))
        output_tokens = len(enc.encode(last_msg_content))

        # Compute cost
        model_name = body.get("model", "unknown-model")
        total_cost = self.cost_calculator.calculate_costs(
            model=model_name,
            input_tokens=self.input_tokens,
            output_tokens=output_tokens,
            compensation=self.valves.compensation,
        )

        # Save cost
        user_email = None
        if __user__:
            user_email = __user__.get("email")
        elif "user" in body:
            user_email = body["user"]

        if user_email:
            try:
                self.user_cost_manager.update_user_cost(
                    user_email, model_name, self.input_tokens, output_tokens, total_cost
                )
            except Exception as e:
                debug_print(f"Error updating user cost: {e}")

        total_tokens = self.input_tokens + output_tokens
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

        stats_list = []
        if self.valves.show_elapsed_time:
            stats_list.append(f"{elapsed:.2f} s")
        if self.valves.show_tokens_per_second:
            stats_list.append(f"{tokens_per_sec:.2f} T/s")
        if self.valves.show_tokens:
            stats_list.append(f"{total_tokens} Tokens")

        # format cost
        cost_str = f"${total_cost:.6f}"
        stats_list.append(cost_str)

        stats_string = " | ".join(stats_list)

        if __event_emitter__:
            await __event_emitter__(
                {"type": "status", "data": {"description": stats_string, "done": True}}
            )

        return body
