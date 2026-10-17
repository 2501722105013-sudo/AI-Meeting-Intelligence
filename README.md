# AI Meeting Intelligence & Decision Tracker 🎙️

An end-to-end local AI-powered meeting assistant built using advanced Natural Language Processing (NLP), deep learning transformers, speech recognition, and vector similarity search.

The system analyzes corporate meeting transcripts (`.txt`, `.pdf`, `.docx`) and audio recordings (`.mp3`, `.wav`) to automatically extract executive summaries, key decisions, action items, named entities, speaker analytics, topic distributions, and provides a vector database AI memory engine to answer queries like *"What was discussed about marketing last month?"*.

---

## 🌟 Key Features

1. **Dual Input Engine**:
   - **Text File Uploads**: Parses `.txt`, `.pdf`, and `.docx` formatted meeting transcripts.
   - **Audio Processing Engine**: Local Speech-to-Text conversion using OpenAI `Whisper` (`.mp3`, `.wav`). No external APIs needed.

2. **Synthetic Dataset Generation**:
   - Generator script (`dataset/generate_dataset.py`) creating **2,000 realistic enterprise meeting transcripts** across 6 corporate domains:
     - Software Development
     - Project Planning
     - Marketing
     - HR
     - Finance
     - Product Management
   - Realistic corporate meeting dialogue with interruptions, consensus building, task assignments, and decision confirmations.
   - Pinned 70% Train, 15% Validation, and 15% Test dataset splits.

3. **8-Module NLP Pipeline**:
   - **Module 1: Preprocessing**: Noise removal, sentence segmentation, tokenization, stopword removal, lemmatization, and speaker turn parsing (NLTK & spaCy).
   - **Module 2: Summarization**: Abstractive summarization via BART / T5 transformers (`sshleifer/distilbart-cnn-12-6`) with TF-IDF keyphrase extractive fallback.
   - **Module 3: Named Entity Recognition**: spaCy & BERT NER for PERSON, DATE, ORGANIZATION, and TASK extraction.
   - **Module 4: Action Item Extraction**: Converts sentences into structured JSON format (`task`, `assigned_person`, `deadline`, `priority`).
   - **Module 5: Decision Extraction**: Sentence classifier + syntactic keyword heuristic detector (`decided`, `approved`, `finalized`, `agreed`, `confirmed`).
   - **Module 6: Topic Modeling**: Dual LDA (scikit-learn) and BERTopic / Sentence-Transformers embedding clustering.
   - **Module 7: AI Meeting Memory / Similarity Search**: Sentence-BERT (`all-MiniLM-L6-v2`) + **FAISS** vector database for natural language query retrieval over historical meetings.
   - **Module 8: Local Audio Transcription**: Local Whisper speech recognition model.

4. **White & Blue Enterprise Dashboard**:
   - Styled with custom CSS (`assets/style.css`) using a corporate White & Blue palette (#1E3A8A deep navy header, #2563EB electric blue accents, #F8FAFC page background).
   - Dynamic Productivity Scoring engine (0-100 score based on decision density, task assignment clarity, and speaker participation entropy).

5. **Exploratory Data Analysis (EDA)**:
   - Interactive Plotly visualizations for meeting duration distribution, speaker contribution breakdown, topic frequency, decision trends, and task assignment matrices.

---

## 📁 Folder Structure

```
Nlp Project/
├── dataset/
│   ├── generate_dataset.py         # Script generating 2,000 realistic meeting transcripts
│   ├── train.json                  # 70% Training split (1,400 meetings)
│   ├── val.json                    # 15% Validation split (300 meetings)
│   ├── test.json                   # 15% Testing split (300 meetings)
│   ├── full_dataset.csv            # Tabular dataset representation for EDA
│   └── sample_transcripts/         # Sample transcript text files for test upload
├── src/
│   ├── __init__.py
│   ├── audio_processor.py          # Local Whisper audio transcription (.mp3, .wav)
│   ├── file_parser.py              # File parsers (.txt, .pdf, .docx)
│   ├── preprocessing.py            # Text cleaning, NLTK/spaCy tokenization & speaker turns
│   ├── summarization.py            # BART/T5 transformer summarization module
│   ├── ner_extraction.py           # spaCy & BERT NER (Person, Date, Org, Task)
│   ├── action_extraction.py        # Structured JSON action item extractor
│   ├── decision_extraction.py      # Decision extraction module
│   ├── topic_modeling.py           # LDA & BERTopic topic modeler
│   ├── vector_search.py            # Sentence-BERT + FAISS AI memory search engine
│   ├── productivity.py             # Meeting productivity scoring calculator
│   ├── database.py                 # SQLite database manager (meetings.db)
│   └── eda.py                      # Interactive Plotly & Matplotlib EDA charts
├── assets/
│   └── style.css                   # Custom White & Blue Streamlit theme stylesheet
├── tests/
│   ├── __init__.py
│   └── test_nlp_pipeline.py        # Automated test suite
├── app.py                          # Streamlit Enterprise Application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python**: 3.9, 3.10, or 3.11
- **FFmpeg**: Required for audio processing with Whisper (Optional if processing text transcripts only).

### 1. Clone & Setup Virtual Environment
```bash
# Navigate to directory
cd "Nlp Project"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download spaCy & NLTK Resources
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 📊 Synthetic Dataset Generation

To generate the dataset of **2,000 realistic enterprise meeting transcripts**:

```bash
python dataset/generate_dataset.py
```

This will create:
- `dataset/train.json` (1,400 records)
- `dataset/val.json` (300 records)
- `dataset/test.json` (300 records)
- `dataset/full_dataset.csv` (2,000 rows for tabular EDA)
- `dataset/sample_transcripts/sample_software_meeting.txt` (Sample for UI testing)

---

## 🚀 Running the Application

Launch the Streamlit enterprise dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the automated test suite to verify all NLP modules, database persistence, and vector search:

```bash
pytest tests/test_nlp_pipeline.py -v
```

---

## 🔒 Offline Execution & Privacy

This application **does not use external LLM APIs** (No OpenAI API key, No Gemini API key, No Claude API key). All models (Whisper, BART/T5, spaCy, Sentence-BERT, FAISS) run **100% locally** on your machine.
