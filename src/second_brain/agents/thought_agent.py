from pydantic_ai import Agent
from second_brain.agents.ingestor import RAGManager
from second_brain.agents.memory_manager import MemoryManager
import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", 'google-gla:gemini-2.5-pro')

system_prompt = """
You are an intelligent reasoning agent within a Second Brain system.

Your primary goals are:
- Retrieve and synthesize information from the ingested knowledge base.
- Use past memory to understand user intent and provide contextual responses.
- Be concise, professional, and accurate.
- Never fabricate information — say “insufficient data” if unsure.
- Avoid decorative formatting unless explicitly asked.
"""

class ThoughtAgent:
    def __init__(self):
        self.agent = Agent(model=LLM_MODEL, system_prompt=system_prompt)
        self.memory = MemoryManager()
        self.rag_manager = RAGManager()

    def run(self, user_prompt: str):
        """Handles RAG retrieval, memory recall, reasoning, and memory storage."""
        print("\n🔍 Retrieving relevant context from knowledge base...")
        rag_context = self.rag_manager.rag_retrieve(user_prompt)

        print("\n🧠 Fetching past memory context...")
        past_memory = self.memory.get_recent_context()

        if past_memory:
            memory_context = "\n".join(
                [f"User: {m['query']}\nAgent: {m['response']}" for m in past_memory]
            )
        else:
            memory_context = "No previous memory yet."

        combined_input = f"""
        Memory Context:
        {memory_context}

        Knowledge Context (RAG):
        {rag_context}

        User Query:
        {user_prompt}
        """

        print("\n🤔 Thinking based on memory and retrieved knowledge...\n")
        response = self.agent.run_sync(combined_input)
        answer = response.output

        # Store new memory
        self.memory.add_entry(user_prompt, answer)

        return answer

    def clear_memory(self):
        self.memory.clear()
        print("🧠 Memory cleared successfully.")
