from pydantic_ai import Agent
from second_brain.agents.ingestor import RAGManager
from second_brain.agents.memory_manager import MemoryManager
from second_brain.utils import get_tracer
from opentelemetry.trace import Status, StatusCode
import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", 'google-gla:gemini-2.5-pro')
tracer = get_tracer("second_brain.thought_agent")

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
        with tracer.start_as_current_span("thought_agent.run") as span:
            span.set_attribute("user_prompt", user_prompt[:100])  # Limit length
            
            try:
                print("\n🔍 Retrieving relevant context from knowledge base...")
                with tracer.start_as_current_span("rag_retrieval") as rag_span:
                    rag_context = self.rag_manager.rag_retrieve(user_prompt)
                    rag_span.set_attribute("context_length", len(rag_context))

                print("\n🧠 Fetching past memory context...")
                with tracer.start_as_current_span("memory_recall") as memory_span:
                    past_memory = self.memory.get_recent_context()
                    memory_span.set_attribute("memory_entries_count", len(past_memory))

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
                with tracer.start_as_current_span("llm_inference") as llm_span:
                    llm_span.set_attribute("model", LLM_MODEL)
                    response = self.agent.run_sync(combined_input)
                    answer = response.output
                    llm_span.set_attribute("response_length", len(answer))

                # Store new memory
                with tracer.start_as_current_span("memory_store") as store_span:
                    self.memory.add_entry(user_prompt, answer)
                    store_span.set_attribute("memory_stored", True)

                span.set_status(Status(StatusCode.OK))
                span.set_attribute("response_length", len(answer))
                return answer
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def clear_memory(self):
        self.memory.clear()
        print("🧠 Memory cleared successfully.")
