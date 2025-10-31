import json
from pathlib import Path

# Store memory JSON inside /memory folder
MEMORY_FILE = Path(__file__).resolve().parents[1] / "memory" / "memory_data.json"

class MemoryManager:
    def __init__(self):
        self.memory = []
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r") as f:
                try:
                    self.memory = json.load(f)
                except json.JSONDecodeError:
                    self.memory = []

    def add_entry(self, user_query: str, response: str):
        """Add a user query and agent response to memory."""
        self.memory.append({
            "query": user_query,
            "response": response
        })
        self._save()

    def get_recent_context(self, n: int = 3):
        """Return the last n entries for context."""
        return self.memory[-n:]

    def clear(self):
        """Reset memory (useful for testing)."""
        self.memory = []
        self._save()

    def _save(self):
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)
