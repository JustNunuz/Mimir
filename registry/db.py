import sqlite3
import json
from pathlib import Path
from datetime import datetime

class RegistryDB:
    def __init__(self, db_path="mimir_registry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    phash TEXT,
                    dhash TEXT,
                    ahash TEXT,
                    watermark_id TEXT,
                    metadata_json TEXT,
                    ai_score REAL,
                    human_score REAL,
                    final_assessment TEXT,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def add_record(self, filename, phash, dhash, ahash, watermark_id, metadata, ai_score, human_score, final_assessment):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO images (
                    filename, phash, dhash, ahash, watermark_id, 
                    metadata_json, ai_score, human_score, final_assessment, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename, phash, dhash, ahash, watermark_id,
                json.dumps(metadata) if metadata else "{}",
                ai_score, human_score, final_assessment, datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def find_by_hash(self, hash_type, hash_value):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT * FROM images WHERE {hash_type} = ?"
            cursor.execute(query, (hash_value,))
            columns = [column[0] for column in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
    
    def get_all_records(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images ORDER BY timestamp DESC")
            columns = [column[0] for column in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
