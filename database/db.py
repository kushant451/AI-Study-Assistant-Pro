import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "study_assistant.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            topic TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER,
            feedback TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents_processed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            word_count INTEGER,
            chunk_count INTEGER
        )
    """)

    conn.commit()
    conn.close()