from pydantic_ai import Agent
from second_brain.agents.ingestor import rag_retrieve
import os
from dotenv import load_dotenv

# Load environment variables (for Gemini or other models)
load_dotenv()

# --- Model & System Prompt ---

LLM_MODEL = os.getenv('LLM_MODEL')

system_prompt = """
You are an intelligent reasoning agent within a Second Brain system.

Your primary goals are:
- Retrieve and synthesize information from the ingested knowledge base.
- Provide concise, accurate, and context-aware responses.
- Maintain a professional, calm, and thoughtful tone.
- When referencing facts or tips, prioritize clarity over verbosity.
- Never fabricate data — if unsure, clearly state that context is insufficient.
- Avoid unnecessary greetings, markdown, or decorative formatting unless explicitly requested.
"""

# --- Initialize Thought Agent ---
thought_agent = Agent(
    model=LLM_MODEL,
    system_prompt=system_prompt,
)

def run_thought_agent(prompt: str):
    """
    Executes a reasoning query with RAG (Retrieval-Augmented Generation).
    Retrieves relevant context from the local ChromaDB and uses it to enhance reasoning.
    """
    print("\n🔍 Retrieving relevant context from knowledge base...")
    context = rag_retrieve(prompt)

    combined_input = f"""
    Relevant Context:
    {context}

    User Query:
    {prompt}
    """

    print("\n🤔 Thinking based on retrieved knowledge...\n")
    response = thought_agent.run_sync(combined_input)

    print("🧠 Thought Agent Response:\n")
    print(response.output)
    print("\n" + "=" * 60 + "\n")

    return response.output
