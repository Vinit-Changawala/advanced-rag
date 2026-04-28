# ============================================================
# reasoning_engine/planner.py
#
# PURPOSE: Break a complex user query into smaller subtasks.
#
# BEGINNER CONCEPT - Why do we need a Planner?
# Simple question: "What is 2+2?" → No planning needed, just answer.
# Complex question: "Compare our Q3 revenue with competitors and
#                   explain what caused the difference."
# → This needs MULTIPLE steps:
#   Step 1: Find our Q3 revenue data
#   Step 2: Find competitor revenue data
#   Step 3: Calculate the difference
#   Step 4: Find documents explaining causes
#   Step 5: Synthesize all of this into an answer
#
# The Planner figures out these steps automatically.
# ============================================================

import json
import logging
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Planner:
    """
    Decomposes user queries into executable subtasks.

    The Planner uses an LLM to analyze the query and create a
    step-by-step plan. Each step is a "task" that other parts
    of the system can execute.

    Usage:
        planner = Planner(llm_client=openai_client)
        plan = planner.create_plan("What caused revenue to drop in Q3?")
        for step in plan["steps"]:
            print(step["action"], step["description"])
    """

    PLANNING_PROMPT = """You are a planning assistant for a RAG system that searches uploaded documents.

CRITICAL RULE: The knowledge base contains uploaded documents (PDFs, papers, CSVs, images).
ALL questions about document content MUST use vector_search as the FIRST step.
Only use sql_query if the question is about database statistics like "how many answers were stored" or "what was the average score".

Available actions:
- "vector_search": Search uploaded documents — USE THIS FIRST for almost all questions
- "sql_query": Query system stats only (answer counts, scores, latency logs)
- "synthesize": Write the final answer from retrieved information — always last
- "validate": Optional check — only include for complex multi-step questions

Return ONLY valid JSON with this structure:
{{
  "complexity": "simple|moderate|complex",
  "requires_agents": false,
  "steps": [
    {{
      "step_number": 1,
      "action": "vector_search",
      "description": "Search for information about the query topic",
      "search_query": "{query}",
      "depends_on": []
    }},
    {{
      "step_number": 2,
      "action": "synthesize",
      "description": "Write answer from retrieved chunks",
      "search_query": "",
      "depends_on": [1]
    }}
  ],
  "estimated_chunks_needed": 5
}}

User query: {query}

JSON plan:"""

    def __init__(self, llm_client=None, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model

    def create_plan(self, query: str,
                    conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Create an execution plan for answering the given query.

        Args:
            query: The user's question
            conversation_history: Previous Q&A pairs for context

        Returns:
            A plan dict with "steps", "complexity", "requires_agents"
        """
        logger.info(f"Creating plan for query: {query[:100]}...")

        if self.llm_client:
            try:
                return self._llm_plan(query, conversation_history)
            except Exception as e:
                logger.warning(f"LLM planning failed, using default plan: {e}")

        # Fallback: simple default plan
        return self._default_plan(query)

    def _llm_plan(self, query: str,
                  history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Use LLM to create an intelligent plan."""
        messages = [
            {
                "role": "system",
                "content": "You are a planning assistant. Always respond with valid JSON only."
            }
        ]

        # Add conversation history for context
        if history:
            for turn in history[-3:]:  # Last 3 turns only (keep prompt short)
                messages.append({"role": "user", "content": turn.get("query", "")})
                messages.append({"role": "assistant", "content": turn.get("answer", "")})

        messages.append({
            "role": "user",
            "content": self.PLANNING_PROMPT.format(query=query)
        })

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.1,         # Very low: we want consistent, logical plans
            response_format={"type": "json_object"}   # Force JSON output
        )

        raw_json = response.choices[0].message.content
        plan = json.loads(raw_json)

        # Add a unique plan ID
        plan["plan_id"] = str(uuid.uuid4())
        plan["original_query"] = query

        logger.info(
            f"Plan created: complexity={plan.get('complexity')}, "
            f"steps={len(plan.get('steps', []))}, "
            f"requires_agents={plan.get('requires_agents')}"
        )

        return plan

    def _default_plan(self, query: str) -> Dict[str, Any]:
        """
        Fallback plan when LLM is not available.
        Creates a simple 2-step plan: search then synthesize.
        """
        return {
            "plan_id": str(uuid.uuid4()),
            "original_query": query,
            "complexity": "simple",
            "requires_agents": False,
            "estimated_chunks_needed": 3,
            "steps": [
                {
                    "step_number": 1,
                    "action": "vector_search",
                    "description": "Search knowledge base for relevant information",
                    "search_query": query,
                    "depends_on": []
                },
                {
                    "step_number": 2,
                    "action": "synthesize",
                    "description": "Generate answer from retrieved chunks",
                    "search_query": "",
                    "depends_on": [1]
                }
            ]
        }
