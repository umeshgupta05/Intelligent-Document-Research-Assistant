import json
from pathlib import Path


class ChatMemory:
    def __init__(self, path="chat_memory.json"):
        self.path = Path(path)
        self.messages = []
        if self.path.exists():
            self.messages = json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, role, content, source=None):
        self.messages.append({"role": role, "content": content, "source": source})
        self.save()

    def as_text(self):
        recent = self.messages[-6:]
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    def clear(self):
        self.messages = []
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")
