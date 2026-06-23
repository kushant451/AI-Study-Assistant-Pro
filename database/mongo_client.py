import os
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

_client = None
_db = None
_connection_checked = False
_connection_ok = False


def _connect():
    global _client, _db, _connection_checked, _connection_ok

    if _connection_checked:
        return _connection_ok

    _connection_checked = True
    uri = os.getenv("MONGODB_URI")

    if not uri:
        try:
            uri = st.secrets["MONGODB_URI"]
        except Exception:
            uri = None

    if not uri:
        _connection_ok = False
        return False

    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client.get_database("study_assistant")
        _connection_ok = True
        return True
    except (ConnectionFailure, ConfigurationError, Exception):
        _connection_ok = False
        return False


def mongo_available():
    return _connect()


def get_db():
    _connect()
    return _db