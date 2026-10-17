"""
Module 1: Text Preprocessing for AI Meeting Intelligence.
Handles noise removal, sentence segmentation, tokenization, stopword removal,
lemmatization, and speaker turn separation using NLTK and spaCy.
"""

import re
import string

# Built-in English stopwords fallback if NLTK data is not pre-downloaded
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", 
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", 
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", 
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", 
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", 
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", 
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", 
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", 
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", 
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    
    # Try downloading required NLTK resources silently
    for pkg in ["punkt", "stopwords", "wordnet"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    NLTK_AVAILABLE = False
    LEMMATIZER = None

try:
    import spacy
    try:
        NLP_SPACY = spacy.load("en_core_web_sm")
    except Exception:
        NLP_SPACY = None
except Exception:
    NLP_SPACY = None


def clean_text(raw_text):
    """Remove noise, fix extra whitespace, and normalize transcript text."""
    if not raw_text:
        return ""
    # Normalize line breaks
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean special artifacts / non-printable characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Normalize multiple spaces (keep linebreaks)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split("\n")]
    return "\n".join([line for line in lines if line])


def parse_speaker_turns(transcript_text):
    """
    Separates speaker turns from a transcript.
    Looks for patterns like 'John:', '[John]', 'John (PM):', etc.
    Returns list of dicts: [{'speaker': 'John', 'text': '...', 'line_num': 1}]
    """
    cleaned = clean_text(transcript_text)
    lines = cleaned.split("\n")
    
    turns = []
    current_speaker = "Unknown Speaker"
    
    speaker_pattern = re.compile(r'^(?:\[?([A-Z][a-zA-B0-9_\s\.\(\)\-]{1,25})\]?:|\(([A-Z][a-zA-B0-9_\s\.\(\)\-]{1,25})\))\s*(.*)$')
    
    for idx, line in enumerate(lines, 1):
        match = speaker_pattern.match(line)
        if match:
            spk = match.group(1) or match.group(2)
            content = match.group(3).strip()
            current_speaker = spk.strip()
            if content:
                turns.append({
                    "speaker": current_speaker,
                    "text": content,
                    "line_num": idx
                })
        else:
            # Inline check for 'SpeakerName: Text'
            parts = line.split(":", 1)
            if len(parts) == 2 and len(parts[0].strip()) <= 30 and not parts[0].strip().startswith("http"):
                current_speaker = parts[0].strip()
                turns.append({
                    "speaker": current_speaker,
                    "text": parts[1].strip(),
                    "line_num": idx
                })
            else:
                turns.append({
                    "speaker": current_speaker,
                    "text": line.strip(),
                    "line_num": idx
                })
                
    return turns


def segment_sentences(text):
    """Segment text into individual sentences using spaCy, NLTK, or regex."""
    if not text:
        return []
        
    if NLP_SPACY:
        doc = NLP_SPACY(text)
        sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if sents:
            return sents
            
    if NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
            
    # Regex fallback sentence segmentation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_and_preprocess(text, lower=True, remove_stopwords=True, lemmatize=True):
    """
    Full text preprocessing pipeline:
    Tokenization, Stopword removal, Lemmatization.
    Returns list of cleaned token strings.
    """
    if not text:
        return []
        
    if NLP_SPACY:
        doc = NLP_SPACY(text)
        tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            word = token.text.lower() if lower else token.text
            if remove_stopwords and (token.is_stop or word in STOPWORDS):
                continue
            if lemmatize:
                word = token.lemma_.lower() if lower else token.lemma_
            tokens.append(word)
        return tokens

    # NLTK / Regex fallback
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower() if lower else text)
    processed = []
    for w in words:
        if remove_stopwords and w.lower() in STOPWORDS:
            continue
        if lemmatize and LEMMATIZER:
            w = LEMMATIZER.lemmatize(w)
        processed.append(w)
        
    return processed


def preprocess_transcript(raw_transcript):
    """
    Complete Preprocessing Pipeline Object returned for downstream NLP modules.
    """
    cleaned = clean_text(raw_transcript)
    turns = parse_speaker_turns(cleaned)
    sentences = []
    for turn in turns:
        sents = segment_sentences(turn["text"])
        for s in sents:
            sentences.append({
                "speaker": turn["speaker"],
                "sentence": s,
                "tokens": tokenize_and_preprocess(s)
            })
            
    speakers_set = list(set([t["speaker"] for t in turns if t["speaker"] != "Unknown Speaker"]))
    
    return {
        "raw_text": raw_transcript,
        "cleaned_text": cleaned,
        "speaker_turns": turns,
        "sentences": sentences,
        "speakers": speakers_set,
        "all_tokens": tokenize_and_preprocess(cleaned)
    }
