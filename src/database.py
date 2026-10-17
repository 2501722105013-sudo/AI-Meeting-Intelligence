"""
Database Manager for AI Meeting Intelligence.
Uses SQLite (meetings.db) for storing processed meetings, decisions, action items,
topic models, and speaker analytics.
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meetings.db")

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Meetings Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            meeting_id TEXT PRIMARY KEY,
            date TEXT,
            department TEXT,
            summary TEXT,
            raw_transcript TEXT,
            duration TEXT,
            speaker_count INTEGER,
            productivity_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 2. Decisions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            decision_text TEXT,
            date TEXT,
            person TEXT,
            FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
        )
        """)

        # 3. Action Items Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            task TEXT,
            assigned_person TEXT,
            deadline TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Pending',
            FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
        )
        """)

        # 4. Topics Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            topic_name TEXT,
            top_words TEXT,
            FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
        )
        """)

        # 5. Speaker Stats Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS speaker_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT,
            speaker_name TEXT,
            sentence_count INTEGER,
            percentage REAL,
            FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
        )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def save_meeting(self, meeting_data: Dict[str, Any]):
        """
        Save complete processed meeting object into SQLite.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        meeting_id = meeting_data.get("meeting_id")
        date_str = meeting_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        dept = meeting_data.get("department", "General")
        summary_val = meeting_data.get("summary", "")
        if isinstance(summary_val, dict):
            summary_text = summary_val.get("summary_text", "")
        else:
            summary_text = str(summary_val)

        raw_transcript = meeting_data.get("raw_transcript", meeting_data.get("transcript", ""))
        duration = meeting_data.get("duration", meeting_data.get("timestamp", "30 minutes"))
        speaker_count = len(meeting_data.get("speaker_names", meeting_data.get("speakers", [])))
        prod_score = float(meeting_data.get("productivity_score", 75.0))

        # Insert or Replace Meeting Record
        cursor.execute("""
        INSERT OR REPLACE INTO meetings (meeting_id, date, department, summary, raw_transcript, duration, speaker_count, productivity_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (meeting_id, date_str, dept, summary_text, raw_transcript, duration, speaker_count, prod_score))

        # Delete old child records if updating
        cursor.execute("DELETE FROM decisions WHERE meeting_id = ?", (meeting_id,))
        cursor.execute("DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,))
        cursor.execute("DELETE FROM topics WHERE meeting_id = ?", (meeting_id,))
        cursor.execute("DELETE FROM speaker_stats WHERE meeting_id = ?", (meeting_id,))

        # Insert Decisions
        for d in meeting_data.get("decisions_made", meeting_data.get("decisions", [])):
            d_text = d.get("decision", "") if isinstance(d, dict) else str(d)
            d_date = d.get("date", "Upcoming") if isinstance(d, dict) else "Upcoming"
            d_person = d.get("person", "Team") if isinstance(d, dict) else "Team"
            cursor.execute("INSERT INTO decisions (meeting_id, decision_text, date, person) VALUES (?, ?, ?, ?)",
                           (meeting_id, d_text, d_date, d_person))

        # Insert Action Items
        for t in meeting_data.get("tasks_assigned", meeting_data.get("action_items", [])):
            task_desc = t.get("task", "")
            assignee = t.get("assigned_person", "Unassigned")
            dl = t.get("deadline", "TBD")
            prio = t.get("priority", "Medium")
            stat = t.get("status", "Pending")
            cursor.execute("INSERT INTO action_items (meeting_id, task, assigned_person, deadline, priority, status) VALUES (?, ?, ?, ?, ?, ?)",
                           (meeting_id, task_desc, assignee, dl, prio, stat))

        # Insert Topics
        for top in meeting_data.get("topics", meeting_data.get("topics_discussed", [])):
            if isinstance(top, dict):
                t_name = top.get("topic_name", "")
                t_words = ", ".join(top.get("top_words", []))
            else:
                t_name = str(top)
                t_words = str(top)
            cursor.execute("INSERT INTO topics (meeting_id, topic_name, top_words) VALUES (?, ?, ?)",
                           (meeting_id, t_name, t_words))

        # Insert Speaker Stats
        for spk, stats in meeting_data.get("speaker_analytics", {}).items():
            s_count = stats.get("count", 0)
            s_pct = stats.get("percentage", 0.0)
            cursor.execute("INSERT INTO speaker_stats (meeting_id, speaker_name, sentence_count, percentage) VALUES (?, ?, ?, ?)",
                           (meeting_id, spk, s_count, s_pct))

        conn.commit()
        conn.close()

    def fetch_all_meetings(self) -> List[Dict[str, Any]]:
        """Fetch summary list of all stored meetings."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meetings ORDER BY created_at DESC")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def fetch_all_decisions(self) -> List[Dict[str, Any]]:
        """Fetch all decisions across meetings."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT d.id, d.meeting_id, d.decision_text as decision, d.date, d.person, m.department, m.date as meeting_date
        FROM decisions d
        JOIN meetings m ON d.meeting_id = m.meeting_id
        ORDER BY m.created_at DESC
        """)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def fetch_all_action_items(self) -> List[Dict[str, Any]]:
        """Fetch all action items across meetings."""
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
        SELECT a.id, a.meeting_id, a.task, a.assigned_person, a.deadline, a.priority, a.status, m.department
        FROM action_items a
        JOIN meetings m ON a.meeting_id = m.meeting_id
        ORDER BY a.priority DESC
        """)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

# Singleton helper
_db_manager = None

def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
