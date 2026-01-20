import os
import pickle
from typing import List, Tuple

MEMORY_DIR = './data/memory'
os.makedirs(MEMORY_DIR, exist_ok=True)


def _mem_path(session_id: str) -> str:
    return os.path.join(MEMORY_DIR, f"{session_id}.pkl")


def load_memory(session_id: str) -> List[Tuple[str, str]]:
    path = _mem_path(session_id)
    if not os.path.exists(path):
        return []
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def save_memory(session_id: str, history: List[Tuple[str, str]]):
    path = _mem_path(session_id)
    with open(path, 'wb') as fh:
        pickle.dump(history, fh)


def append_memory(session_id: str, question: str, answer: str, max_turns: int = 10):
    history = load_memory(session_id)
    history.append((question, answer))
    history = history[-max_turns:]
    save_memory(session_id, history)
    return history
