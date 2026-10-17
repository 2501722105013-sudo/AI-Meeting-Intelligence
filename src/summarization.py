"""
Module 2: Automatic Meeting Summarization for AI Meeting Intelligence.
Generates concise executive meeting summaries covering main discussion points,
key decisions, and outcomes using local BART / T5 transformers,
with an extractive fallback mechanism.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MeetingSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize local summarization pipeline.
        Model options: 'sshleifer/distilbart-cnn-12-6', 'facebook/bart-large-cnn', 't5-small'.
        """
        self.model_name = model_name
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from transformers import pipeline
            logger.info(f"Loading local summarization model '{self.model_name}'...")
            self.pipeline = pipeline("summarization", model=self.model_name, tokenizer=self.model_name)
        except Exception as e:
            logger.warning(f"Could not load HuggingFace summarization model ({e}). Using Extractive Fallback.")
            self.pipeline = None

    def summarize(self, transcript_text: str, max_length=150, min_length=40) -> Dict[str, Any]:
        """
        Generate structured summary object from transcript text.
        Returns:
        {
          "summary_text": "...",
          "main_points": ["...", "..."],
          "key_outcomes": ["...", "..."],
          "method": "BART/T5" or "Extractive"
        }
        """
        if not transcript_text or len(transcript_text.strip()) == 0:
            return {
                "summary_text": "No transcript content provided.",
                "main_points": [],
                "key_outcomes": [],
                "method": "None"
            }

        # Pre-clean transcript text for summarization input
        lines = [line.strip() for line in transcript_text.split("\n") if line.strip()]
        clean_text_corpus = " ".join(lines)
        
        summary_str = ""
        method_used = "Extractive (TF-IDF Keyphrase)"

        if self.pipeline is not None and len(clean_text_corpus.split()) > 30:
            try:
                # Chunk long text if needed
                max_chunk = 1000
                words = clean_text_corpus.split()
                if len(words) > max_chunk:
                    text_input = " ".join(words[:max_chunk])
                else:
                    text_input = clean_text_corpus

                res = self.pipeline(text_input, max_length=max_length, min_length=min_length, do_sample=False)
                summary_str = res[0]['summary_text']
                method_used = f"Abstractive ({self.model_name})"
            except Exception as e:
                logger.warning(f"Abstractive summarization failed: {e}. Falling back to extractive.")
                summary_str = self._extractive_summary(lines)
        else:
            summary_str = self._extractive_summary(lines)

        # Structure summary points
        structured = self._format_structured_summary(summary_str, lines)
        structured["method"] = method_used
        return structured

    def _extractive_summary(self, lines: List[str], num_sentences=4) -> str:
        """
        Fallback TF-IDF weighted sentence selection for fast, reliable local summarization.
        """
        from collections import Counter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', " ".join(lines).lower())
        word_freq = Counter(words)
        
        sentence_scores = []
        for line in lines:
            # Remove speaker tag for scoring
            body = line.split(":", 1)[-1] if ":" in line else line
            s_words = re.findall(r'\b[a-zA-Z]{3,}\b', body.lower())
            score = sum([word_freq.get(w, 0) for w in s_words])
            sentence_scores.append((score, line))

        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        top_lines = [item[1] for item in sentence_scores[:num_sentences]]
        
        return " ".join(top_lines)

    def _format_structured_summary(self, summary_text: str, raw_lines: List[str]) -> Dict[str, Any]:
        """Format the summary into Main Discussion Points and Key Outcomes."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
        
        main_points = []
        key_outcomes = []

        for s in sentences:
            if any(kw in s.lower() for kw in ["decid", "agree", "complet", "launch", "approv", "will", "finish"]):
                key_outcomes.append(s)
            else:
                main_points.append(s)

        if not main_points:
            main_points = sentences[:2]
        if not key_outcomes:
            key_outcomes = sentences[2:] if len(sentences) > 2 else [sentences[-1]] if sentences else []

        return {
            "summary_text": summary_text,
            "main_points": main_points,
            "key_outcomes": key_outcomes
        }

# Global singleton helper
_summarizer = None

def generate_meeting_summary(transcript_text: str) -> Dict[str, Any]:
    global _summarizer
    if _summarizer is None:
        _summarizer = MeetingSummarizer()
    return _summarizer.summarize(transcript_text)
