# ============================================================
# multi_agent_system/base_agent.py
#
# PURPOSE: Define the common interface all agents must follow.
#
# BEGINNER CONCEPT - Why a Base Class?
# All agents (Research, Synthesis, Critique) share common behavior:
# - They all accept tasks
# - They all return results
# - They all have memory of previous steps
# - They all log what they're doing
#
# Instead of writing this code 3 times, we write it ONCE in
# BaseAgent, then each agent "inherits" it.
# Inheritance: child class gets all methods from parent class.
# ============================================================

import logging
import uuid
from abc import ABC, abstractmethod   # ABC = Abstract Base Class
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class that all agents must extend.

    BEGINNER CONCEPT - Abstract Class:
    A class marked ABC cannot be used directly.
    You MUST create a subclass that implements all @abstractmethod methods.
    This enforces a contract: "all agents MUST have a run() method."

    Usage (you extend this, never use it directly):
        class MyAgent(BaseAgent):
            def run(self, task, context):
                # implement your logic here
                return {"result": "..."}
    """

    def __init__(self, name: str, llm_client=None, vector_store=None):
        """
        Args:
            name: Human-readable agent name (e.g., "Research Agent")
            llm_client: OpenAI client for generating responses
            vector_store: For agents that need to search documents
        """
        self.name = name
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.agent_id = str(uuid.uuid4())

        # Memory: list of past tasks this agent has seen in this session
        # This gives agents short-term memory within a single query
        self.memory: List[Dict] = []

        logger.info(f"Agent initialized: {self.name} (id={self.agent_id[:8]})")

    @abstractmethod
    def run(self, task: Dict[str, Any],
            context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        This MUST be implemented by every subclass.
        The @abstractmethod decorator enforces this.

        Args:
            task: What the agent should do
                  e.g., {"instruction": "Find information about X"}
            context: Shared context (query, previous results, chunks)

        Returns:
            Result dict with at minimum {"success": bool, "output": any}
        """
        pass   # 'pass' because abstract methods have no body

    def remember(self, task: Dict, result: Dict):
        """
        Store a task-result pair in memory.
        Agents use this to avoid repeating work.
        """
        self.memory.append({"task": task, "result": result})

        # Keep only the last 10 memories to avoid unbounded growth
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]

    def get_memory_summary(self) -> str:
        """Return a brief summary of recent memory for use in prompts."""
        if not self.memory:
            return "No previous steps."

        lines = []
        for i, item in enumerate(self.memory[-3:]):   # Last 3 steps
            task_desc = item["task"].get("instruction", "")[:80]
            success = item["result"].get("success", False)
            lines.append(f"Step {i+1}: {task_desc} → {'OK' if success else 'FAILED'}")

        return "\n".join(lines)

    def _call_llm(self, prompt: str, system_prompt: str = "",
                  temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """
        Helper method to call the LLM.
        All agents use this so we don't repeat the API call code.
        """
        if not self.llm_client:
            raise RuntimeError(f"{self.name}: LLM client not configured")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' id='{self.agent_id[:8]}'>"
