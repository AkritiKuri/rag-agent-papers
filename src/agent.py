"""
agent.py
A lightweight router/agent. Given a user query, an LLM first decides
which ACTION to take (search / summarize / compare), then that action
is executed against the RAG pipeline. This is the piece that makes the
system "agentic" rather than a single fixed retrieve-then-answer path.
"""
import json
import re
import os
from typing import Literal

from langchain_ollama import ChatOllama
from src.rag_core import RAGPipeline

ROUTER_SYSTEM_PROMPT = """You are a router for a research-paper Q&A system. \
Given the user's message, decide which single action to take:

- "search": a factual question answerable by retrieving relevant chunks \
from the corpus (default for most questions).
- "summarize": the user wants a summary of one specific paper. Extract the \
paper name/title as best you can into "target_paper".
- "compare": the user wants two (or more) papers compared. Extract the \
paper names into "target_papers" as a list.

Respond ONLY with compact JSON, no prose, no markdown fences, in this shape:
{"action": "search" | "summarize" | "compare",
 "target_paper": string or null,
 "target_papers": list of strings or null,
 "query": the original or reformulated question}
"""


class ResearchAgent:
    def __init__(self, rag: RAGPipeline, router_model: str = "phi3:mini"):
        self.rag = rag
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.router = ChatOllama(model=router_model, temperature=0, base_url=ollama_base_url)

    def _route(self, user_message: str) -> dict:
        messages = [
            ("system", ROUTER_SYSTEM_PROMPT),
            ("human", user_message),
        ]
        response = self.router.invoke(messages)
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # Smaller local models sometimes add stray prose around the JSON,
        # so pull out the first {...} block rather than requiring the
        # whole response to be pure JSON.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fall back to plain search if the router misbehaves
            return {"action": "search", "target_paper": None,
                    "target_papers": None, "query": user_message}

    def _do_search(self, plan: dict) -> dict:
        result = self.rag.answer(plan["query"])
        return {"action": "search", **result}

    def _do_summarize(self, plan: dict) -> dict:
        target = plan.get("target_paper") or plan["query"]
        query = f"Provide a thorough summary of the paper: {target}"
        result = self.rag.answer(query, k=8)
        return {"action": "summarize", "target_paper": target, **result}

    def _do_compare(self, plan: dict) -> dict:
        targets = plan.get("target_papers") or [plan["query"]]
        query = (
            f"Compare these papers on their methodology, findings, and "
            f"limitations: {', '.join(targets)}"
        )
        result = self.rag.answer(query, k=10)
        return {"action": "compare", "target_papers": targets, **result}

    def handle(self, user_message: str) -> dict:
        plan = self._route(user_message)
        action = plan.get("action", "search")
        if action == "summarize":
            return self._do_summarize(plan)
        if action == "compare":
            return self._do_compare(plan)
        return self._do_search(plan)