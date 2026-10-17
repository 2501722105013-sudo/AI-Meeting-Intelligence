"""
Module 3: Named Entity Recognition (NER) for AI Meeting Intelligence.
Extracts PERSON, DATE, ORGANIZATION, and TASK entities using spaCy and BERT/transformers.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MeetingNERExtractor:
    def __init__(self):
        self.nlp_spacy = None
        self.bert_ner = None
        self._init_models()

    def _init_models(self):
        try:
            import spacy
            try:
                self.nlp_spacy = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp_spacy = None
        except Exception:
            self.nlp_spacy = None

        try:
            from transformers import pipeline
            self.bert_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        except Exception:
            self.bert_ner = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from meeting transcript text.
        Returns dict with keys: PERSON, DATE, ORGANIZATION, TASK
        """
        persons = set()
        dates = set()
        organizations = set()
        tasks = set()

        if not text or not text.strip():
            return {"PERSON": [], "DATE": [], "ORGANIZATION": [], "TASK": []}

        # 1. spaCy NER Extraction
        if self.nlp_spacy:
            doc = self.nlp_spacy(text)
            for ent in doc.ents:
                val = ent.text.strip()
                if len(val) < 2:
                    continue
                if ent.label_ == "PERSON":
                    persons.add(val)
                elif ent.label_ in ["DATE", "TIME"]:
                    dates.add(val)
                elif ent.label_ in ["ORG", "NORP"]:
                    organizations.add(val)

        # 2. BERT NER Extraction (for additional coverage)
        if self.bert_ner:
            try:
                # Process line by line to avoid token length overflow
                for line in text.split("\n")[:20]:
                    if not line.strip():
                        continue
                    ner_results = self.bert_ner(line[:512])
                    for item in ner_results:
                        entity_group = item.get("entity_group", "")
                        word = item.get("word", "").strip()
                        if entity_group == "PER":
                            persons.add(word)
                        elif entity_group == "ORG":
                            organizations.add(word)
            except Exception as e:
                logger.warning(f"BERT NER execution error: {e}")

        # 3. Rule-based & Syntactic Task Extraction
        task_patterns = [
            r'(?:will|needs to|must|should|going to|assigned to)\s+([a-z0-9\s]{5,40}?)(?:\bby\b|\bon\b|\bnext\b|\buntil\b|\.|\n)',
            r'(?:task:|action:)\s*([^\n\.]+)'
        ]
        for pattern in task_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                clean_t = m.strip().capitalize()
                if len(clean_t) > 4:
                    tasks.add(clean_t)

        # 4. Supplemental Rule-based Date Extraction (e.g. "Monday", "August 20", "next week")
        date_pattern = r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow|yesterday|next week|end of month|q[1-4]|january|february|march|april|may|june|july|august|september|october|november|december)\b'
        rule_dates = re.findall(date_pattern, text, re.IGNORECASE)
        for rd in rule_dates:
            dates.add(rd.capitalize())

        return {
            "PERSON": sorted(list(persons)),
            "DATE": sorted(list(dates)),
            "ORGANIZATION": sorted(list(organizations)),
            "TASK": sorted(list(tasks))
        }

    def format_ner_output(self, text: str) -> str:
        """Formatted human-readable NER representation."""
        entities = self.extract_entities(text)
        formatted = []
        if entities["PERSON"]:
            formatted.append(f"Person:\n{', '.join(entities['PERSON'])}")
        if entities["DATE"]:
            formatted.append(f"Deadline / Date:\n{', '.join(entities['DATE'])}")
        if entities["ORGANIZATION"]:
            formatted.append(f"Organization:\n{', '.join(entities['ORGANIZATION'])}")
        if entities["TASK"]:
            formatted.append(f"Task:\n{', '.join(entities['TASK'])}")
        return "\n\n".join(formatted) if formatted else "No entities detected."

# Module helper
_ner_extractor = None

def extract_meeting_ner(text: str) -> Dict[str, List[str]]:
    global _ner_extractor
    if _ner_extractor is None:
        _ner_extractor = MeetingNERExtractor()
    return _ner_extractor.extract_entities(text)
