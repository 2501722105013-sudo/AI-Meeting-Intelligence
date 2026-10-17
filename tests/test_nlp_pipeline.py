"""
Automated Integration and Unit Test Suite for AI Meeting Intelligence.
Tests all 8 NLP modules, file parsers, database persistence, and vector search.
"""

import pytest
import os
import json
import sys

# Ensure root folder is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_parser import parse_txt
from src.preprocessing import preprocess_transcript, parse_speaker_turns
from src.summarization import generate_meeting_summary
from src.ner_extraction import extract_meeting_ner
from src.action_extraction import extract_action_items_list
from src.decision_extraction import extract_meeting_decisions
from src.topic_modeling import get_meeting_topics
from src.vector_search import get_vector_store
from src.productivity import calculate_productivity_score
from src.database import DatabaseManager

SAMPLE_TRANSCRIPT = (
    "Alex: Welcome to the software architecture review sync.\n"
    "Sarah: Thanks Alex. I will complete the microservice API migration by Friday.\n"
    "David: Great. We decided to approve the cloud database indexing proposal.\n"
    "Alex: Excellent. Priya will schedule the QA security audit by next Monday."
)

def test_file_parser():
    parsed = parse_txt(SAMPLE_TRANSCRIPT)
    assert "software architecture" in parsed

def test_preprocessing():
    prep = preprocess_transcript(SAMPLE_TRANSCRIPT)
    assert len(prep["speaker_turns"]) >= 4
    assert "Alex" in prep["speakers"]
    assert len(prep["sentences"]) >= 4

def test_summarization():
    summary = generate_meeting_summary(SAMPLE_TRANSCRIPT)
    assert "summary_text" in summary
    assert len(summary["summary_text"]) > 0

def test_ner_extraction():
    ner = extract_meeting_ner(SAMPLE_TRANSCRIPT)
    assert "PERSON" in ner
    assert "DATE" in ner
    assert "TASK" in ner

def test_action_item_extraction():
    actions = extract_action_items_list(SAMPLE_TRANSCRIPT)
    assert len(actions) > 0
    item = actions[0]
    assert "task" in item
    assert "assigned_person" in item
    assert "deadline" in item
    assert "priority" in item

def test_decision_extraction():
    decisions = extract_meeting_decisions(SAMPLE_TRANSCRIPT)
    assert len(decisions) > 0
    dec = decisions[0]
    assert "decision" in dec
    assert "cloud database" in dec["decision"].lower() or "approve" in dec["decision"].lower()

def test_topic_modeling():
    sentences = ["We discussed API microservice architecture.", "Cloud database indexing is key.", "Security audit next week."]
    topics = get_meeting_topics(sentences, method="LDA")
    assert len(topics) > 0

def test_vector_search():
    vstore = get_vector_store()
    meeting_data = {
        "meeting_id": "TEST_001",
        "date": "2025-08-20",
        "department": "Software Development",
        "summary": "Discussed microservice API migration and cloud database indexing.",
        "topics": ["API Migration", "Cloud Database"],
        "decisions": [{"decision": "Approved cloud database indexing", "date": "Next Monday", "person": "David"}]
    }
    vstore.add_meeting(meeting_data)
    results = vstore.search_meetings("cloud database indexing", top_k=1)
    assert len(results) > 0
    assert results[0]["meeting_id"] == "TEST_001"

def test_productivity_score():
    decisions = [{"decision": "Approved launch"}]
    tasks = [{"task": "Complete testing", "assigned_person": "Sarah", "deadline": "Friday"}]
    turns = [{"speaker": "Alex"}, {"speaker": "Sarah"}]
    prod = calculate_productivity_score(decisions, tasks, turns)
    assert "overall_score" in prod
    assert prod["overall_score"] > 0

def test_database_persistence(tmp_path):
    db_file = os.path.join(tmp_path, "test_meetings.db")
    db_mgr = DatabaseManager(db_path=db_file)
    m_data = {
        "meeting_id": "DB_TEST_001",
        "date": "2025-08-20",
        "department": "Marketing",
        "summary": "Marketing campaign strategy",
        "raw_transcript": SAMPLE_TRANSCRIPT,
        "decisions_made": [{"decision": "Approved campaign budget", "date": "Friday", "person": "Alex"}],
        "tasks_assigned": [{"task": "Draft ad copy", "assigned_person": "Sarah", "deadline": "Friday", "priority": "High", "status": "Pending"}],
        "topics_discussed": ["Ad Campaign", "Budget"],
        "speaker_analytics": {"Alex": {"count": 2, "percentage": 50.0}, "Sarah": {"count": 2, "percentage": 50.0}}
    }
    db_mgr.save_meeting(m_data)
    meetings = db_mgr.fetch_all_meetings()
    assert len(meetings) == 1
    assert meetings[0]["meeting_id"] == "DB_TEST_001"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
