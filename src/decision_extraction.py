"""
Module 5: Decision Extraction for AI Meeting Intelligence.
Identifies and extracts decision sentences from transcripts using keyword heuristics,
syntactic pattern matching, and sentence classification.
Keywords detected: decided, approved, finalized, agreed, confirmed, moving forward with.
"""

import re
from typing import List, Dict, Any

DECISION_KEYWORDS = [
    "decided", "approved", "finalized", "agreed", "confirmed",
    "decide", "approve", "finalize", "agree", "confirm",
    "moving forward with", "resolved", "consensus"
]

class DecisionExtractor:
    def __init__(self):
        pass

    def extract_decisions(self, transcript_text: str) -> List[Dict[str, str]]:
        """
        Extract decisions from transcript text.
        Returns list of dicts:
        [
          {
            "decision": "Product launch approved for next month",
            "date": "Next month",
            "person": "John"
          }
        ]
        """
        decisions = []
        if not transcript_text:
            return decisions

        lines = [line.strip() for line in transcript_text.split("\n") if line.strip()]
        
        seen_decisions = set()

        for line in lines:
            speaker = "Team"
            line_body = line
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts[0].strip()) <= 25:
                    speaker = parts[0].strip()
                    line_body = parts[1].strip()

            # Check if sentence contains decision indicators
            s_lower = line_body.lower()
            if any(kw in s_lower for kw in DECISION_KEYWORDS):
                clean_decision = self._format_decision_text(line_body)
                if clean_decision and clean_decision.lower() not in seen_decisions:
                    seen_decisions.add(clean_decision.lower())
                    date_val = self._extract_date_mention(line_body)
                    decisions.append({
                        "decision": clean_decision,
                        "date": date_val,
                        "person": speaker
                    })

        # Heuristic fallback if no explicit keyword found
        if not decisions:
            for line in lines:
                if any(phrase in line.lower() for phrase in ["we will ", "let's go with ", "our plan is "]):
                    clean_decision = self._format_decision_text(line)
                    if clean_decision:
                        decisions.append({
                            "decision": clean_decision,
                            "date": "Next Sync",
                            "person": line.split(":")[0].strip() if ":" in line else "Team"
                        })
                        break

        return decisions

    def _format_decision_text(self, text: str) -> str:
        # Remove initial speaker tag or filler text
        t = re.sub(r'^(?:[A-Z][a-z]+:\s*)', '', text)
        t = re.sub(r'^(?:so|well|okay|thanks|great|looks like)\s*,?\s*', '', t, flags=re.IGNORECASE)
        t = t.strip(" .,;:-")
        return t.capitalize() if t else ""

    def _extract_date_mention(self, text: str) -> str:
        date_pattern = r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|next week|next month|by end of month|august \d+|q[1-4])\b'
        match = re.search(date_pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).capitalize()
        return "Upcoming Quarter"

# Module helper
_decision_extractor = None

def extract_meeting_decisions(transcript_text: str) -> List[Dict[str, str]]:
    global _decision_extractor
    if _decision_extractor is None:
        _decision_extractor = DecisionExtractor()
    return _decision_extractor.extract_decisions(transcript_text)
