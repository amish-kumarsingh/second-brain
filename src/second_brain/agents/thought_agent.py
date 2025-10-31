from pydantic_ai import Agent
from second_brain.agents.ingestor import RAGManager
from second_brain.agents.memory_manager import MemoryManager
from second_brain.utils import get_tracer, get_guard
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
        self.pii_guard = get_guard()

    def run(self, user_prompt: str):
        """Handles RAG retrieval, memory recall, reasoning, and memory storage.
        All inputs and outputs are sanitized for PII before processing.
        """
        with tracer.start_as_current_span("thought_agent.run") as span:
            # Sanitize user prompt first
            sanitized_prompt = self.pii_guard.sanitize(user_prompt)
            span.set_attribute("user_prompt", sanitized_prompt[:100])  # Limit length
            span.set_attribute("pii_guard_enabled", self.pii_guard.enabled)
            
            try:
                print("\n🔍 Retrieving relevant context from knowledge base...")
                with tracer.start_as_current_span("rag_retrieval") as rag_span:
                    # RAG retrieval uses sanitized prompt
                    rag_context = self.rag_manager.rag_retrieve(sanitized_prompt)
                    # Sanitize RAG context as it may contain PII from stored documents
                    rag_context = self.pii_guard.sanitize(rag_context)
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
                    
                    # Memory context is already sanitized (stored sanitized), but sanitize again for safety
                    memory_context = self.pii_guard.sanitize(memory_context)

                # All inputs are sanitized before sending to LLM
                combined_input = f"""
                Memory Context:
                {memory_context}

                Knowledge Context (RAG):
                {rag_context}

                User Query:
                {sanitized_prompt}
                """

                print("\n🤔 Thinking based on memory and retrieved knowledge...\n")
                with tracer.start_as_current_span("llm_inference") as llm_span:
                    llm_span.set_attribute("model", LLM_MODEL)
                    response = self.agent.run_sync(combined_input)
                    answer = response.output
                    # Sanitize LLM output before returning (LLM might include PII)
                    answer = self.pii_guard.sanitize(answer)
                    llm_span.set_attribute("response_length", len(answer))

                # Store new memory (memory manager will sanitize again, but that's okay)
                with tracer.start_as_current_span("memory_store") as store_span:
                    # Use original user_prompt for storage (will be sanitized by memory manager)
                    # This ensures we store what the user actually said (sanitized)
                    self.memory.add_entry(sanitized_prompt, answer)
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
