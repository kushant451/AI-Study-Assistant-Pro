import os
import json
from datetime import datetime
from database.mongo_client import get_db, mongo_available

_FALLBACK_DIR = os.path.join(os.path.dirname(__file__), "_local_chat_history")


def _fallback_path(username):
    os.makedirs(_FALLBACK_DIR, exist_ok=True)
    safe_name = "".join(c for c in username if c.isalnum() or c in "_-")
    return os.path.join(_FALLBACK_DIR, f"{safe_name}.json")


def save_message(username, role, content, tool=None, extra=None):
    message = {
        "username": username,
        "role": role,
        "content": content,
        "tool": tool,
        "extra": extra,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if mongo_available():
        db = get_db()
        db.chat_history.insert_one(message)
    else:
        path = _fallback_path(username)
        history = []
        if os.path.exists(path):
            with open(path, "r") as f:
                history = json.load(f)
        history.append(message)
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)


def load_history(username, limit=100):
    if mongo_available():
        db = get_db()
        cursor = (
            db.chat_history.find({"username": username})
            .sort("timestamp", -1)
            .limit(limit)
        )
        messages = list(cursor)
        messages.reverse()
        for m in messages:
            m["_id"] = str(m["_id"])
        return messages
    else:
        path = _fallback_path(username)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            history = json.load(f)
        return history[-limit:]


def clear_history(username):
    if mongo_available():
        db = get_db()
        db.chat_history.delete_many({"username": username})
    else:
        path = _fallback_path(username)
        if os.path.exists(path):
            os.remove(path)