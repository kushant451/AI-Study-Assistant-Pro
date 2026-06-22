from datetime import datetime
from auth.jwt_auth import hash_password, verify_password, create_access_token
from database.mongo_client import get_db, mongo_available

import json
import os

_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "_local_users.json")


def _load_fallback_users():
    if not os.path.exists(_FALLBACK_PATH):
        return {}
    with open(_FALLBACK_PATH, "r") as f:
        return json.load(f)


def _save_fallback_users(users):
    with open(_FALLBACK_PATH, "w") as f:
        json.dump(users, f, indent=2)


def signup(username, password):
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    password_hash = hash_password(password)

    if mongo_available():
        db = get_db()
        existing = db.users.find_one({"username": username})
        if existing:
            return False, "Username already exists."

        db.users.insert_one({
            "username": username,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
        })
        return True, "Account created successfully."
    else:
        users = _load_fallback_users()
        if username in users:
            return False, "Username already exists."
        users[username] = {
            "password_hash": password_hash,
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_fallback_users(users)
        return True, "Account created successfully (local storage - configure MongoDB for production use)."


def login(username, password):
    print("Login called")

    available = mongo_available()
    print("Mongo available:", available)

    if available:
        db = get_db()
        print("DB object:", db)

        user = db.users.find_one({"username": username})
        print("User found:", user is not None)

        if not user:
            return False, "Invalid username or password."

        if not verify_password(password, user["password_hash"]):
            return False, "Invalid username or password."

        token = create_access_token(user["_id"], username)
        return True, token
    else:
        print("Using fallback storage")

        users = _load_fallback_users()
        user = users.get(username)

        if not user:
            return False, "Invalid username or password."

        if not verify_password(password, user["password_hash"]):
            return False, "Invalid username or password."

        token = create_access_token(username, username)
        return True, token

def get_current_user(token):
    from auth.jwt_auth import decode_access_token
    payload = decode_access_token(token)
    if payload:
        return payload.get("username")
    return None