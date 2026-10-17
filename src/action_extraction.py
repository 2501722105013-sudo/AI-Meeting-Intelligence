"""
Module 4: Action Item Extraction for AI Meeting Intelligence.
Parses meeting sentences into structured action item JSON objects:
{
  "task": "...",
  "assigned_person": "...",
  "deadline": "...",
  "priority": "High / Medium / Low"
}
"""

import re
import json
from typing import List, Dict, Any

DAYS_AND_TIMELINES = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "tomorrow", "next week", "end of month", "end of quarter", "by friday",
    "by monday", "asap", "end of day", "eod"
]

HIGH_PRIORITY_KEYWORDS = ["urgent", "critical", "high", "asap", "immediately", "security", "blocker", "p0", "p1"]
LOW_PRIORITY_KEYWORDS = ["optional", "low", "eventually", "nice to have", "backlog", "future"]

class ActionItemExtractor:
    def __init__(self):
        pass

    def extract_action_items(self, transcript_text: str, speaker_turns: List[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """
        Extract structured action items from transcript text or list of speaker turns.
        """
        action_items = []
        if not transcript_text:
            return action_items

        lines = transcript_text.split("\n")
        
        # Regex patterns for matching task assignments
        patterns = [
            # "Sarah will complete the website testing by Friday."
            r'([A-Z][a-z]+)\s+(?:will|is going to|needs to|must|should|to)\s+([a-z0-9\s,]{4,60}?)(?:\bby\b|\bon\b|\bbefore\b|\buntil\b)\s+([a-zA-Z0-9\s]{3,20})(?:\.|\n|$)',
            # "John: I will handle the deployment by end of week."
            r'([A-Z][a-z]+):\s*(?:I|we)\s+(?:will|shall|need to)\s+([a-z0-9\s,]{4,60}?)(?:\bby\b|\bon\b|\bbefore\b)\s+([a-zA-Z0-9\s]{3,20})(?:\.|\n|$)',
            # "Task: Complete website testing assigned to Sarah by Friday"
            r'(?:task|action):\s*([a-z0-9\s,]{4,60}?)\s+(?:assigned to|for)\s+([A-Z][a-z]+)(?:\s+by\s+([a-zA-Z0-9\s]{3,20}))?'
        ]

        seen_tasks = set()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Extract speaker if available
            speaker = "Unassigned"
            if ":" in line_str:
                potential_spk, body = line_str.split(":", 1)
                if len(potential_spk.strip()) <= 25:
                    speaker = potential_spk.strip()
                    line_text = body.strip()
                else:
                    line_text = line_str
            else:
                line_text = line_str

            matched = False
            for p_idx, p in enumerate(patterns):
                match = re.search(p, line_text, re.IGNORECASE)
                if match:
                    if p_idx == 0:
                        person = match.group(1).capitalize()
                        task_desc = match.group(2).strip()
                        deadline = match.group(3).strip()
                    elif p_idx == 1:
                        person = match.group(1).capitalize()
                        task_desc = match.group(2).strip()
                        deadline = match.group(3).strip()
                    else:
                        task_desc = match.group(1).strip()
                        person = match.group(2).capitalize()
                        deadline = match.group(3).strip() if match.group(3) else "TBD"

                    task_clean = self._clean_task_desc(task_desc)
                    if task_clean and task_clean.lower() not in seen_tasks:
                        seen_tasks.add(task_clean.lower())
                        priority = self._determine_priority(line_text)
                        action_items.append({
                            "task": task_clean,
                            "assigned_person": person,
                            "deadline": self._clean_deadline(deadline),
                            "priority": priority,
                            "status": "Pending"
                        })
                        matched = True
                        break

            # Fallback heuristic scan if explicit pattern did not match
            if not matched and any(modal in line_text.lower() for modal in [" will ", " need to ", " assigned to ", " handle "]):
                task_cand = self._extract_fallback_task(line_text, speaker)
                if task_cand and task_cand["task"].lower() not in seen_tasks:
                    seen_tasks.add(task_cand["task"].lower())
                    action_items.append(task_cand)

        return action_items

    def _clean_task_desc(self, task_str: str) -> str:
        t = re.sub(r'^(?:to|and|the|a|an)\s+', '', task_str.strip(), flags=re.IGNORECASE)
        t = t.strip(" .,;:-")
        return t.capitalize() if t else ""

    def _clean_deadline(self, deadline_str: str) -> str:
        d = deadline_str.strip(" .,;:-")
        return d.capitalize() if d else "TBD"

    def _determine_priority(self, sentence: str) -> str:
        s_low = sentence.lower()
        if any(kw in s_low for kw in HIGH_PRIORITY_KEYWORDS):
            return "High"
        elif any(kw in s_low for kw in LOW_PRIORITY_KEYWORDS):
            return "Low"
        return "Medium"

    def _extract_fallback_task(self, line_text: str, default_speaker: str) -> Dict[str, str]:
        # Extract potential person
        person = default_speaker
        spk_match = re.search(r'\b([A-Z][a-z]+)\b', line_text)
        if spk_match and spk_match.group(1) not in ["I", "We", "The", "Let"]:
            person = spk_match.group(1)

        # Extract potential deadline
        deadline = "Next Sync"
        for d in DAYS_AND_TIMELINES:
            if d in line_text.lower():
                deadline = d.capitalize()
                break

        # Remove speaker tag and common filler
        clean_s = line_text.split(":", 1)[-1] if ":" in line_text else line_text
        clean_s = re.sub(r'(?:I|we|[A-Z][a-z]+)\s+(?:will|need to|should)\s+', '', clean_s, flags=re.IGNORECASE)
        
        task_desc = clean_s.strip(" .,;:-")
        if len(task_desc) > 5:
            return {
                "task": task_desc.capitalize()[:80],
                "assigned_person": person,
                "deadline": deadline,
                "priority": self._determine_priority(line_text),
                "status": "Pending"
            }
        return None

# Module helper
_action_extractor = None

def extract_action_items_json(transcript_text: str) -> str:
    global _action_extractor
    if _action_extractor is None:
        _action_extractor = ActionItemExtractor()
    items = _action_extractor.extract_action_items(transcript_text)
    return json.dumps(items, indent=2)

def extract_action_items_list(transcript_text: str) -> List[Dict[str, str]]:
    global _action_extractor
    if _action_extractor is None:
        _action_extractor = ActionItemExtractor()
    return _action_extractor.extract_action_items(transcript_text)
